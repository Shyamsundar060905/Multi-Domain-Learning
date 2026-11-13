"""Multi-dataset Mobile-CDNet: one network shared by every dataset, plus
per-dataset parts that can each be switched on:

  domain_bn      every BatchNorm in the encoder (features[0..17]) and in the
                 decoder becomes one BatchNorm per dataset (its own scale,
                 shift and running statistics), each starting as an exact copy
                 of the original (ImageNet) BatchNorm.            51 + 12 layers
  enc_adapters   a residual adapter per dataset on each of the 5 encoder
                 outputs the decoder uses (features[1], [3], [6], [13], [17]).
                 Applied in line: the adapted feature also feeds the next
                 encoder block.                                       5 adapters
  diff_adapters  a residual adapter per dataset on each of the 5 difference
                 maps |date-1 features - date-2 features|.            5 adapters
  dec_adapters   a residual adapter per dataset after each of the decoder's
                 12 conv-BN-ReLU layers.                             12 adapters

Everything else -- every conv weight and the final classifier -- is shared.

Optional second task, ``recon``: a reconstruction decoder rebuilds each input
image from that date's deepest encoder features (features[17], the same
tensor the change path takes its deepest difference from).  It is a side
branch: nothing flows from it into the change path, so the change map is
computed exactly as without it; it only shapes the shared encoder through its
loss, and it is not needed at test time.  Same design as the
bottleneck-reconstruction branch's ReconstructionDecoder (5 stride-2
transposed convs + BatchNorm + ReLU, a 3x3 output conv per date), run once per
date with the same weights.  Its BatchNorms are per dataset when domain_bn is
on.

* With every switch off the network IS models.model.BaseNet: same modules,
  same parameter names, same outputs.
* Every adapter's last 1x1 conv starts at zero, so with switches on the
  untrained network still gives BaseNet's outputs.
* Creating the adapters or the reconstruction decoder does not change the
  random-number stream, so every switch combination starts from the same
  shared weights and (same seed, same datasets) trains on the same batches in
  the same order.
tools/check_md.py checks all of this.
"""
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BaseNet

# Encoder outputs the decoder uses (the indices MobileNetV2.forward collects)
# and their channels (= the decoder's in_d).
TAPS = (1, 3, 6, 13, 17)
CHANNELS = (16, 24, 32, 96, 320)
PARTS = ('domain_bn', 'enc_adapters', 'diff_adapters', 'dec_adapters')
# How many per-dataset modules each part must create (fail loudly otherwise).
PART_MODULES = {'domain_bn': 63, 'enc_adapters': 5, 'diff_adapters': 5, 'dec_adapters': 12}
# Reconstruction decoder stage widths (as on the bottleneck-reconstruction branch).
RECON_WIDTHS = (256, 128, 64, 32, 16)


class ResidualAdapter(nn.Module):
    """1x1 down -> BN -> ReLU -> 1x1 up (zero-init) -> dropout, added to the input.

    Same design as the adapters of the U-Net multi-domain models.
    """

    def __init__(self, channels, reduction=16, dropout=0.1):
        super(ResidualAdapter, self).__init__()
        bottleneck = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.bn(self.down(x)))))


class ReconstructionDecoder(nn.Module):
    """Rebuilds one date's image from that date's deepest encoder features.

    As the bottleneck-reconstruction branch's ReconstructionDecoder: each stage
    is a 2x2 stride-2 transposed conv (no bias) -> BatchNorm -> ReLU, so 5
    stages take 1/32 size back to full size; then a 3x3 conv to RGB, with a
    separate output conv for each date.  PyTorch's default initialisation.
    """

    def __init__(self, in_ch, widths=RECON_WIDTHS, out_channels=3):
        super(ReconstructionDecoder, self).__init__()
        self.ups = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = in_ch
        for w in widths:
            self.ups.append(nn.ConvTranspose2d(ch, w, kernel_size=2, stride=2, bias=False))
            self.norms.append(nn.BatchNorm2d(w))
            ch = w
        self.act = nn.ReLU(inplace=True)
        self.head_t1 = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        self.head_t2 = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, date):
        if date not in ('t1', 't2'):
            raise ValueError("date must be 't1' or 't2', got %r" % (date,))
        for up, norm in zip(self.ups, self.norms):
            x = self.act(norm(up(x)))
        return (self.head_t1 if date == 't1' else self.head_t2)(x)


class PerDataset(nn.Module):
    """One copy of a module per dataset; runs the active dataset's copy.

    MultiDatasetNet.forward sets the active dataset before anything runs, so
    these can sit inside the original nn.Sequential blocks.
    """

    def __init__(self, copies):
        super(PerDataset, self).__init__()
        self.per_dataset = nn.ModuleDict(copies)
        self.active = None

    def forward(self, x):
        if self.active is None:
            raise RuntimeError('PerDataset module used before a dataset was set')
        return self.per_dataset[self.active](x)


def _adapter(channels, datasets, reduction, dropout):
    return PerDataset(OrderedDict(
        (d, ResidualAdapter(channels, reduction, dropout)) for d in datasets))


def _per_dataset_batchnorms(module, datasets):
    """Replace every BatchNorm2d under ``module`` by per-dataset copies of it."""
    made = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            pd = PerDataset(OrderedDict((d, copy.deepcopy(child)) for d in datasets))
            setattr(module, name, pd)
            made.append(pd)
        else:
            made += _per_dataset_batchnorms(child, datasets)
    return made


def _with_adapters(seq, datasets, reduction, dropout):
    """Copy of a conv-BN-ReLU nn.Sequential with a per-dataset adapter after each ReLU."""
    layers, made, out_ch = [], [], None
    for layer in seq.children():
        layers.append(layer)
        if isinstance(layer, nn.Conv2d):
            out_ch = layer.out_channels
        elif isinstance(layer, nn.ReLU):
            ad = _adapter(out_ch, datasets, reduction, dropout)
            layers.append(ad)
            made.append(ad)
    return nn.Sequential(*layers), made


class MultiDatasetNet(nn.Module):
    """Mobile-CDNet (models.model.BaseNet) with optional per-dataset parts.

    ``base``: a BaseNet to build from (its encoder and decoder modules are
    taken over and modified); by default a new BaseNet(3, 1) is made, exactly
    as tools/train.py does.
    """

    def __init__(self, datasets, domain_bn=False, enc_adapters=False, diff_adapters=False,
                 dec_adapters=False, adapter_reduction=16, adapter_dropout=0.1, base=None,
                 recon=False):
        super(MultiDatasetNet, self).__init__()
        datasets = list(datasets)
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError('datasets must be distinct names, got %r' % (datasets,))
        self.datasets = datasets
        if base is None:
            base = BaseNet(3, 1)
        self.backbone = base.backbone      # MobileNetV2, ImageNet weights
        self.swa = base.swa                # NeighborFeatureAggregation decoder
        if tuple(self.swa.in_d) != CHANNELS:
            raise ValueError('decoder channels %s, expected %s' % (self.swa.in_d, CHANNELS))

        self.enc_adapters = None
        self.diff_adapters = None
        self.parts = OrderedDict((p, []) for p in PARTS)
        r, p = adapter_reduction, adapter_dropout

        if domain_bn:
            for i in range(TAPS[-1] + 1):  # features[18] is never used
                self.parts['domain_bn'] += _per_dataset_batchnorms(self.backbone.features[i], datasets)
            self.parts['domain_bn'] += _per_dataset_batchnorms(self.swa, datasets)

        rng = torch.get_rng_state()
        if enc_adapters:
            self.enc_adapters = nn.ModuleList([_adapter(c, datasets, r, p) for c in CHANNELS])
            self.parts['enc_adapters'] = list(self.enc_adapters)
        if diff_adapters:
            self.diff_adapters = nn.ModuleList([_adapter(c, datasets, r, p) for c in CHANNELS])
            self.parts['diff_adapters'] = list(self.diff_adapters)
        if dec_adapters:
            for name in ('conv2d1', 'conv2d2', 'conv2d3', 'conv2d4'):
                seq, made = _with_adapters(getattr(self.swa, name), datasets, r, p)
                setattr(self.swa, name, seq)
                self.parts['dec_adapters'] += made
            for name in ('DoubleConv5', 'cat4', 'cat3', 'cat2'):
                block = getattr(self.swa, name)
                block.double_conv, made = _with_adapters(block.double_conv, datasets, r, p)
                self.parts['dec_adapters'] += made
        self.recon = None
        self.recon_bn = []
        if recon:
            # Same starting decoder whichever adapters are on.
            torch.set_rng_state(rng)
            self.recon = ReconstructionDecoder(CHANNELS[-1])
            if domain_bn:
                self.recon_bn = _per_dataset_batchnorms(self.recon, datasets)
                if len(self.recon_bn) != len(RECON_WIDTHS):
                    raise RuntimeError('reconstruction decoder: made %d per-dataset BatchNorms, expected %d'
                                       % (len(self.recon_bn), len(RECON_WIDTHS)))
        torch.set_rng_state(rng)

        on = {'domain_bn': domain_bn, 'enc_adapters': enc_adapters,
              'diff_adapters': diff_adapters, 'dec_adapters': dec_adapters}
        for part, mods in self.parts.items():
            want = PART_MODULES[part] if on[part] else 0
            if len(mods) != want:
                raise RuntimeError('%s: made %d per-dataset modules, expected %d'
                                   % (part, len(mods), want))
        self.switches = on
        self._per_dataset = [m for mods in self.parts.values() for m in mods] + self.recon_bn

    # ------------------------------------------------------------------
    def set_dataset(self, dataset):
        if dataset not in self.datasets:
            raise ValueError('unknown dataset %r; this model has %s' % (dataset, self.datasets))
        for m in self._per_dataset:
            m.active = dataset

    def encode(self, x):
        """MobileNetV2.forward, with the encoder adapters applied at the 5 outputs."""
        feats = []
        for idx, layer in enumerate(self.backbone.features):
            if idx > TAPS[-1]:
                break  # features[18] runs in MobileNetV2.forward but its output is never used
            x = layer(x)
            if idx in TAPS:
                if self.enc_adapters is not None:
                    x = self.enc_adapters[TAPS.index(idx)](x)
                feats.append(x)
        return feats

    def decode(self, c):
        """NeighborFeatureAggregation.forward, starting from the difference maps."""
        s = self.swa
        c1, c2, c3, c4, c5 = c
        d1 = s.conv2d1(s.downsample(c1))
        d2 = s.conv2d2(c2)
        d3 = s.conv2d3(c3)
        d4 = s.conv2d4(c4)
        d5 = s.DoubleConv5(c5)
        e4 = F.interpolate(d5, scale_factor=(2, 2), mode='bilinear')
        f4 = s.cat4(torch.cat([d4, e4], dim=1))
        e3 = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')
        f3 = s.cat3(torch.cat([d3, e3], dim=1))
        e2 = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')
        s2 = s.cat2(torch.cat([e2, d2, d1], dim=1))
        return s.cls(s2)

    def forward(self, x1, x2, dataset, return_recon=False):
        """Change probability map; with ``return_recon`` also (T1', T2').

        The change map is computed first and the same way either way; the
        reconstruction decoder only reads each date's deepest features.
        """
        self.set_dataset(dataset)
        f1 = self.encode(x1)
        f2 = self.encode(x2)
        c = [torch.abs(a - b) for a, b in zip(f1, f2)]
        if self.diff_adapters is not None:
            c = [ad(ci) for ad, ci in zip(self.diff_adapters, c)]
        mask = self.decode(c)
        mask = F.interpolate(mask, scale_factor=(4, 4), mode='bilinear')
        mask = torch.sigmoid(mask)
        if not return_recon:
            return mask
        if self.recon is None:
            raise RuntimeError('return_recon=True needs a model built with recon=True')
        return mask, (self.recon(f1[-1], 't1'), self.recon(f2[-1], 't2'))

    # ------------------------------------------------------------------
    def dataset_parameters(self, dataset):
        """Parameters that belong to ``dataset`` only."""
        return [q for m in self._per_dataset for q in m.per_dataset[dataset].parameters()]

    def shared_parameters(self):
        own = {id(q) for d in self.datasets for q in self.dataset_parameters(d)}
        return [q for q in self.parameters() if id(q) not in own]

    def part_sizes(self, dataset=None):
        """Parameters of one dataset in each per-dataset part (plus 'recon_bn' with recon)."""
        d = self.datasets[0] if dataset is None else dataset
        groups = list(self.parts.items())
        if self.recon is not None:
            groups.append(('recon_bn', self.recon_bn))
        return OrderedDict((part, sum(q.numel() for m in mods for q in m.per_dataset[d].parameters()))
                           for part, mods in groups)

    def recon_parameter_count(self):
        """All reconstruction-decoder parameters (every dataset's BatchNorm copies included)."""
        return 0 if self.recon is None else sum(q.numel() for q in self.recon.parameters())
