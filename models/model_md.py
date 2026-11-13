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

Optional fusion, ``fusion_stages`` (needs recon), as the bottleneck-
reconstruction branch's reconstruction-change exchange: for each chosen
reconstruction stage k, D_k = |stage-k features of date 1 - of date 2| is
  * concatenated to the input of the change-decoder block at the same size
    (stage 1, 256 ch, 1/16 -> cat4;  2, 128 ch, 1/8 -> cat3;  3, 64 ch,
    1/4 -> cat2); that block's first conv gets the extra input channels,
    with their weights starting at zero, so the untrained change map is
    still BaseNet's;
  * (fusion_heads) fed to its own change head (2x2 up-convs + BN + ReLU to
    full size, then a 1x1 conv; the branch's AuxDecoder) for an extra loss.
With fusion the change map depends on reconstruction stages 1..max(k), so
those are needed at test time too.

* With every switch off the network IS models.model.BaseNet: same modules,
  same parameter names, same outputs.
* Every adapter's last 1x1 conv starts at zero, so with switches on the
  untrained network still gives BaseNet's outputs.
* Creating the adapters, the reconstruction decoder or the fusion parts does
  not change the random-number stream, so every switch combination starts
  from the same shared weights and (same seed, same datasets) trains on the
  same batches in the same order.
tools/check_md.py checks all of this.
"""
import copy
import math
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
# Fusion: reconstruction stage -> change-decoder block at the same size, and
# that block's original input channels (d4+e4, d3+e3, e2+d2+d1).
FUSION_TARGETS = {1: ('cat4', 192), 2: ('cat3', 64), 3: ('cat2', 72)}
# Fusion change-head widths per stage and the prior its output starts at
# (the branch's _RC_HEAD_WIDTHS and prior).
FUSION_HEAD_WIDTHS = {1: (128, 64, 32, 16), 2: (64, 32, 16), 3: (32, 16)}
FUSION_PRIOR = 0.02


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

    def forward(self, x, date, return_feats=False):
        """Rebuilt image; with ``return_feats`` also each stage's output (stage k = feats[k-1])."""
        if date not in ('t1', 't2'):
            raise ValueError("date must be 't1' or 't2', got %r" % (date,))
        feats = []
        for up, norm in zip(self.ups, self.norms):
            x = self.act(norm(up(x)))
            feats.append(x)
        image = (self.head_t1 if date == 't1' else self.head_t2)(x)
        return (image, feats) if return_feats else image


class FusionHead(nn.Module):
    """Change head on one reconstruction stage's difference D_k.

    The bottleneck-reconstruction branch's AuxDecoder: 2x2 stride-2 up-convs
    (no bias) -> BatchNorm -> ReLU back to full size, then a 1x1 conv; that
    last conv starts with weights N(0, 0.01) and bias log(p / (1 - p)),
    p = 0.02, as there.  Returns probabilities (sigmoid), like the main output.
    """

    def __init__(self, in_ch, widths):
        super(FusionHead, self).__init__()
        self.ups = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = in_ch
        for w in widths:
            self.ups.append(nn.ConvTranspose2d(ch, w, kernel_size=2, stride=2, bias=False))
            self.norms.append(nn.BatchNorm2d(w))
            ch = w
        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(ch, 1, kernel_size=1)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(FUSION_PRIOR / (1.0 - FUSION_PRIOR)))

    def forward(self, x):
        for up, norm in zip(self.ups, self.norms):
            x = self.act(norm(up(x)))
        return torch.sigmoid(self.classifier(x))


def _widen_first_conv(seq, old_in, extra):
    """Give the first conv of ``seq`` ``extra`` more input channels, their weights at zero."""
    old = seq[0]
    if not isinstance(old, nn.Conv2d) or old.in_channels != old_in:
        raise RuntimeError('expected a Conv2d with %d input channels first, found %r' % (old_in, old))
    new = nn.Conv2d(old_in + extra, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old_in].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    seq[0] = new
    return extra * old.out_channels * old.kernel_size[0] * old.kernel_size[1]


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
                 recon=False, fusion_stages=(), fusion_heads=True):
        super(MultiDatasetNet, self).__init__()
        datasets = list(datasets)
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError('datasets must be distinct names, got %r' % (datasets,))
        fusion_stages = tuple(sorted(set(int(k) for k in fusion_stages)))
        if fusion_stages and not recon:
            raise ValueError('fusion needs the reconstruction decoder (recon=True)')
        if not set(fusion_stages) <= set(FUSION_TARGETS):
            raise ValueError('fusion stages must be among 1, 2, 3; got %s' % (fusion_stages,))
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
        # Fusion (made after the reconstruction decoder, so that decoder starts the same with or without it).
        self.fusion_stages = fusion_stages
        self.fusion_widened = OrderedDict()   # stage -> extra change-decoder weights
        self.fusion_heads = None
        self.fusion_bn = []
        for k in fusion_stages:
            block, old_in = FUSION_TARGETS[k]
            self.fusion_widened[k] = _widen_first_conv(getattr(self.swa, block).double_conv, old_in,
                                                       RECON_WIDTHS[k - 1])
        if fusion_stages and fusion_heads:
            self.fusion_heads = nn.ModuleDict(OrderedDict(
                (str(k), FusionHead(RECON_WIDTHS[k - 1], FUSION_HEAD_WIDTHS[k])) for k in fusion_stages))
            if domain_bn:
                self.fusion_bn = _per_dataset_batchnorms(self.fusion_heads, datasets)
        torch.set_rng_state(rng)

        on = {'domain_bn': domain_bn, 'enc_adapters': enc_adapters,
              'diff_adapters': diff_adapters, 'dec_adapters': dec_adapters}
        for part, mods in self.parts.items():
            want = PART_MODULES[part] if on[part] else 0
            if len(mods) != want:
                raise RuntimeError('%s: made %d per-dataset modules, expected %d'
                                   % (part, len(mods), want))
        self.switches = on
        self._per_dataset = [m for mods in self.parts.values() for m in mods] + self.recon_bn + self.fusion_bn

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

    def decode(self, c, fused=None):
        """NeighborFeatureAggregation.forward, starting from the difference maps.

        ``fused``: {stage: D_k} from fusion, appended (last) to the input of
        cat4 / cat3 / cat2 for stage 1 / 2 / 3; empty or None = the original.
        """
        fused = fused or {}
        s = self.swa

        def cat(parts, stage):
            if stage in fused:
                if fused[stage].shape[-2:] != parts[0].shape[-2:]:
                    raise ValueError('fusion stage %d is %s, its change-decoder block is %s'
                                     % (stage, tuple(fused[stage].shape[-2:]), tuple(parts[0].shape[-2:])))
                parts = parts + [fused[stage]]
            return torch.cat(parts, dim=1)

        c1, c2, c3, c4, c5 = c
        d1 = s.conv2d1(s.downsample(c1))
        d2 = s.conv2d2(c2)
        d3 = s.conv2d3(c3)
        d4 = s.conv2d4(c4)
        d5 = s.DoubleConv5(c5)
        e4 = F.interpolate(d5, scale_factor=(2, 2), mode='bilinear')
        f4 = s.cat4(cat([d4, e4], 1))
        e3 = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')
        f3 = s.cat3(cat([d3, e3], 2))
        e2 = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')
        s2 = s.cat2(cat([e2, d2, d1], 3))
        return s.cls(s2)

    def forward(self, x1, x2, dataset, return_aux=False):
        """Change probability map; with ``return_aux`` also
        {'recon': (T1', T2'), 'heads': {stage: fusion-head probability map}}.

        Without fusion the reconstruction decoder only reads each date's
        deepest features and is run only for ``return_aux``; with fusion it
        always runs, because the change map uses its stage differences.  (It
        draws no random numbers, so where it runs does not change dropout.)
        """
        self.set_dataset(dataset)
        f1 = self.encode(x1)
        f2 = self.encode(x2)
        c = [torch.abs(a - b) for a, b in zip(f1, f2)]
        if self.diff_adapters is not None:
            c = [ad(ci) for ad, ci in zip(self.diff_adapters, c)]
        if return_aux and self.recon is None:
            raise RuntimeError('return_aux=True needs a model built with recon=True')
        recons, fused, heads = None, {}, OrderedDict()
        if self.recon is not None and (return_aux or self.fusion_stages):
            r1, s1 = self.recon(f1[-1], 't1', return_feats=True)
            r2, s2 = self.recon(f2[-1], 't2', return_feats=True)
            recons = (r1, r2)
            fused = {k: torch.abs(s1[k - 1] - s2[k - 1]) for k in self.fusion_stages}
            if return_aux and self.fusion_heads is not None:
                heads = OrderedDict((k, self.fusion_heads[str(k)](fused[k])) for k in self.fusion_stages)
        mask = self.decode(c, fused)
        mask = F.interpolate(mask, scale_factor=(4, 4), mode='bilinear')
        mask = torch.sigmoid(mask)
        if not return_aux:
            return mask
        return mask, {'recon': recons, 'heads': heads}

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
        if self.fusion_heads is not None:
            groups.append(('fusion_bn', self.fusion_bn))
        return OrderedDict((part, sum(q.numel() for m in mods for q in m.per_dataset[d].parameters()))
                           for part, mods in groups)

    def recon_parameter_count(self):
        """All reconstruction-decoder parameters (every dataset's BatchNorm copies included)."""
        return 0 if self.recon is None else sum(q.numel() for q in self.recon.parameters())

    def fusion_head_parameter_count(self):
        return 0 if self.fusion_heads is None else sum(q.numel() for q in self.fusion_heads.parameters())

    def training_only_parameters(self):
        """Parameters the change map never uses (they only serve the training losses):
        the reconstruction stages after the last fusion stage, the two RGB output
        convs and the fusion heads; without fusion, the whole reconstruction decoder."""
        if self.recon is None:
            return []
        k = max(self.fusion_stages) if self.fusion_stages else 0
        mods = list(self.recon.ups)[k:] + list(self.recon.norms)[k:] + [self.recon.head_t1, self.recon.head_t2]
        if self.fusion_heads is not None:
            mods.append(self.fusion_heads)
        return [q for m in mods for q in m.parameters()]
