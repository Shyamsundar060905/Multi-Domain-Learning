"""Checks for the multi-dataset Mobile-CDNet before any training.  Run on the
DGX (MobileCDNet env) from the repository root; it runs on the CPU and takes
about a minute:

    python tools/check_md.py

1. ImageNet weights: every MobileNetV2 tensor gets an ImageNet value (only
   the unused 1000-class classifier is left over) and BaseNet's encoder holds
   exactly those values.
2. All switches off: same parameter names and count as BaseNet, same output.
3. Per-dataset BatchNorms are exact copies of the originals; with every
   switch on, the untrained network still gives BaseNet's output (each
   adapter starts as the identity); turning parts on does not change the
   shared starting weights or the random stream (so the same batches).
4. Parameters per dataset for each part, and in total.
5. A training step on WHU leaves every LEVIR and SYSU parameter and
   BatchNorm statistic unchanged and trains all of WHU's adapters.  Also shows
   that plain zero_grad() would NOT leave them unchanged.
6. Swapping date 1 and date 2 gives the same output.
7. Reconstruction task: the change map is exactly BaseNet's with the
   reconstruction decoder added (the change path is untouched), every other
   weight and the random stream are unchanged, the reconstruction loss reaches
   the encoder but not the change decoder (and the change loss does not reach
   the reconstruction decoder), parameter counts, per-dataset isolation, and
   the PSNR helper exactly inverts the input normalisation.
8. Fusion: the widened change-decoder conv keeps its original weights and
   starts the new ones at zero (untrained change map = BaseNet's), nothing
   else changes, parameter counts, which losses reach which parts, the
   date-swap symmetry, stages 1-3 with every part on, per-dataset isolation,
   and fusion without reconstruction is refused.
9. tools/train_md.py end to end on tiny synthetic data: two datasets with
   every part on, a refused re-run, --resume, one dataset with every part
   off, one dataset with --recon, and one with --recon and fusion at stage 2;
   the experiment header at the start says what is on and the parameter
   counts.
"""
import contextlib
import copy
import io
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
for path in (TOOLS, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2  # noqa: E402
from torch.hub import load_state_dict_from_url  # noqa: E402

from models.MobileNetV2 import MobileNetV2, model_urls  # noqa: E402
from models.model import BaseNet  # noqa: E402
from models.model_md import MultiDatasetNet  # noqa: E402
import train_md  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    ok = ok and bool(cond)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def count(m):
    return sum(q.numel() for q in m.parameters())


DS = ['LEVIR', 'WHU', 'SYSU']
ALL_ON = dict(domain_bn=True, enc_adapters=True, diff_adapters=True, dec_adapters=True)
torch.manual_seed(0)
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
target = (torch.rand(2, 1, 64, 64) > 0.8).float()


def max_diff(a, b):
    return (a - b).abs().max().item()


def same_output(model, base, dataset, mode):
    m, b = copy.deepcopy(model), copy.deepcopy(base)
    getattr(m, mode)()
    getattr(b, mode)()
    with torch.no_grad():
        return max_diff(m(x1, x2, dataset), b(x1, x2))


print('=== 1. ImageNet weights ===')
sd = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=False)
res = MobileNetV2().load_state_dict(sd, strict=False)
missing = [k for k in res.missing_keys if not k.endswith('num_batches_tracked')]
check(not missing, 'every MobileNetV2 weight has an ImageNet value (missing: %s)' % missing[:5])
check(bool(res.unexpected_keys) and all(k.startswith('classifier.') for k in res.unexpected_keys),
      'only the unused classifier is left over: %s' % res.unexpected_keys)
torch.manual_seed(0)
base = quiet(BaseNet, 3, 1)
bsd = base.backbone.state_dict()
keys = [k for k in bsd if not k.endswith('num_batches_tracked')]
check(all(k in sd and torch.equal(bsd[k], sd[k]) for k in keys),
      "BaseNet's encoder holds exactly the ImageNet values (%d tensors)" % len(keys))
base_total = count(base)
print('  BaseNet parameters: {:,}'.format(base_total))

print('\n=== 2. every switch off = BaseNet ===')
md0 = MultiDatasetNet(['LEVIR'], base=copy.deepcopy(base))
check(list(md0.state_dict()) == list(base.state_dict()), 'same parameter names, same order')
check(count(md0) == base_total, 'same parameter count ({:,})'.format(count(md0)))
for mode in ('eval', 'train'):
    d = same_output(md0, base, 'LEVIR', mode)
    check(d <= 1e-6, '%s mode: same output as BaseNet (max |difference| %.1e)' % (mode, d))

print('\n=== 3. per-dataset parts start as no-ops ===')
md_bn = MultiDatasetNet(DS, domain_bn=True, base=copy.deepcopy(base))
check(len(md_bn.parts['domain_bn']) == 63, '63 BatchNorms made per-dataset (51 encoder + 12 decoder)')
ref = base.state_dict()
copies = [(n, t) for n, t in md_bn.state_dict().items() if '.per_dataset.' in n]
exact = True
for name, t in copies:
    head, rest = name.split('.per_dataset.', 1)
    exact = exact and torch.equal(t, ref[head + '.' + rest.split('.', 1)[1]])
check(exact and len(copies) == 63 * 5 * len(DS),
      'every per-dataset BatchNorm (weights + running statistics) is an exact copy (%d tensors)' % len(copies))
md = MultiDatasetNet(DS, base=copy.deepcopy(base), **ALL_ON)
for mode in ('eval', 'train'):
    worst = max(same_output(md, base, d, mode) for d in DS)
    check(worst <= 1e-6, 'every part on, untrained, %s mode: BaseNet output for all 3 datasets '
          '(max |difference| %.1e)' % (mode, worst))
torch.manual_seed(1)
off = quiet(MultiDatasetNet, DS)
rng_off = torch.get_rng_state()
torch.manual_seed(1)
on = quiet(MultiDatasetNet, DS, **ALL_ON)
rng_on = torch.get_rng_state()
check(torch.equal(rng_off, rng_on), 'parts on or off: same random stream afterwards (same batches)')
check(torch.equal(off.swa.cls.weight, on.swa.cls.weight)
      and torch.equal(off.swa.cat2.double_conv[0].weight, on.swa.cat2.double_conv[0].weight),
      'parts on or off: same starting shared decoder weights')

print('\n=== 4. parameter counts ===')
EXPECT = {'domain_bn': (32800, 63), 'enc_adapters': (15592, 5),
          'diff_adapters': (15592, 5), 'dec_adapters': (10176, 12)}
for part, (per, n_mods) in EXPECT.items():
    m = MultiDatasetNet(DS, base=copy.deepcopy(base), **{part: True})
    sizes = [sum(q.numel() for q in m.dataset_parameters(d)) for d in DS]
    extra = (len(DS) - 1) * per if part == 'domain_bn' else len(DS) * per
    check(len(m.parts[part]) == n_mods and sizes == [per] * len(DS) and count(m) == base_total + extra,
          '{:13s} {} modules, {:,} per dataset, total {:,}'.format(part, len(m.parts[part]), sizes[0], count(m)))
per_all = sum(p for p, _ in EXPECT.values())
sizes = [sum(q.numel() for q in md.dataset_parameters(d)) for d in DS]
shared = sum(q.numel() for q in md.shared_parameters())
check(sizes == [per_all] * len(DS) and per_all == 74160, 'every part on: {:,} per dataset'.format(sizes[0]))
check(count(md) == base_total - 32800 + len(DS) * per_all and shared + sum(sizes) == count(md),
      'every part on, 3 datasets: total {:,} = shared {:,} + 3 x {:,}'.format(count(md), shared, per_all))
ids = [{id(q) for q in md.dataset_parameters(d)} for d in DS]
check(not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]),
      'no parameter belongs to two datasets')

print('\n=== 5. one dataset\'s step leaves the others alone ===')


def one_step(model, opt, dataset, set_to_none):
    model.train()
    loss = train_md.BCEDiceLoss(model(x1, x2, dataset), target)
    opt.zero_grad(set_to_none=set_to_none)
    loss.backward()
    opt.step()


trained = None
for set_to_none in (True, False):
    m = copy.deepcopy(md)
    opt = torch.optim.Adam(m.parameters(), 5e-4, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    one_step(m, opt, 'LEVIR', set_to_none)
    one_step(m, opt, 'SYSU', set_to_none)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    one_step(m, opt, 'WHU', set_to_none)
    after = m.state_dict()
    moved = [k for k in before if ('.per_dataset.LEVIR.' in k or '.per_dataset.SYSU.' in k)
             and not torch.equal(before[k], after[k])]
    if set_to_none:
        trained = m
        check(not moved, 'zero_grad(set_to_none=True): a WHU step changes no LEVIR/SYSU tensor '
              '(%d changed)' % len(moved))
        ups = [k for k in after if '.per_dataset.WHU.' in k and k.endswith('up.weight')]
        check(len(ups) == 22 and all(after[k].abs().sum().item() > 0 for k in ups),
              "a WHU step trains all 22 of WHU's adapters (their zero-initialised last convs moved)")
        shared_moved = any(not torch.equal(before[k], after[k]) for k in after
                           if k.startswith('swa.cls.'))
        check(shared_moved, 'a WHU step trains the shared classifier')
    else:
        print('  INFO  plain zero_grad(): %d LEVIR/SYSU tensors change during a WHU step '
              '(why train_md.py uses set_to_none=True)' % len(moved))

print('\n=== 6. swapping the dates ===')
trained.eval()
with torch.no_grad():
    worst = max(max_diff(trained(x1, x2, d), trained(x2, x1, d)) for d in DS)
check(worst <= 1e-6, 'trained model, all 3 datasets: same output with the dates swapped '
      '(max |difference| %.1e)' % worst)

print('\n=== 7. reconstruction task (--recon) ===')
RECON_SHARED = 501760 + 870   # 5 transposed convs + the two 3x3 output convs
RECON_BN = 992                # BatchNorm weight + bias: 2 x (256 + 128 + 64 + 32 + 16)
mr = MultiDatasetNet(['LEVIR'], recon=True, base=copy.deepcopy(base))
check(count(mr) - base_total == RECON_SHARED + RECON_BN == mr.recon_parameter_count() == 503622,
      'one dataset: the reconstruction decoder adds {:,} parameters'.format(count(mr) - base_total))
check(list(base.state_dict()) == [k for k in mr.state_dict() if not k.startswith('recon.')],
      'every change-path tensor keeps its name and order; the only new ones are recon.*')
for mode in ('eval', 'train'):
    d = same_output(mr, base, 'LEVIR', mode)
    check(d <= 1e-6, '%s mode: change map identical to BaseNet with the reconstruction decoder added '
          '(max |difference| %.1e)' % (mode, d))
mr.eval()
with torch.no_grad():
    plain = mr(x1, x2, 'LEVIR')
    mask_r, aux_r = mr(x1, x2, 'LEVIR', return_aux=True)
    r1, r2 = aux_r['recon']
check(torch.equal(plain, mask_r), 'the change map is the same whether or not the reconstructions are computed')
check(tuple(r1.shape) == tuple(r2.shape) == tuple(x1.shape), "T1' and T2' have the input's shape (2, 3, 64, 64)")

torch.manual_seed(1)
a = quiet(MultiDatasetNet, DS, **ALL_ON)
rng_a = torch.get_rng_state()
torch.manual_seed(1)
b = quiet(MultiDatasetNet, DS, recon=True, **ALL_ON)
rng_b = torch.get_rng_state()
torch.manual_seed(1)
c = quiet(MultiDatasetNet, DS, recon=True)
sa, sb = a.state_dict(), b.state_dict()
check(torch.equal(rng_a, rng_b), 'reconstruction on or off: same random stream afterwards (same batches)')
check(all(torch.equal(sa[k], sb[k]) for k in sa), 'reconstruction on or off: every other weight starts the same')
check(torch.equal(b.recon.ups[0].weight, c.recon.ups[0].weight)
      and torch.equal(b.recon.head_t2.weight, c.recon.head_t2.weight),
      'the reconstruction decoder starts the same whichever other parts are on')


def has_grad(module):
    return any(q.grad is not None and q.grad.abs().sum().item() > 0 for q in module.parameters())


mg = MultiDatasetNet(DS, recon=True, base=copy.deepcopy(base), **ALL_ON)
mg.train()
recons = mg(x1, x2, 'WHU', return_aux=True)[1]['recon']
mg.zero_grad(set_to_none=True)
train_md.recon_mse(recons, x1, x2).backward()
change_side = [q for n, q in mg.named_parameters() if n.startswith('swa.') or n.startswith('diff_adapters.')]
check(all(q.grad is None for q in change_side),
      'the reconstruction loss sends no gradient into the change decoder or the difference adapters')
check(has_grad(mg.backbone.features[0]) and has_grad(mg.backbone.features[17]) and has_grad(mg.recon),
      'the reconstruction loss trains the reconstruction decoder and reaches the shared encoder (first and last block)')
mg.zero_grad(set_to_none=True)
mask_g, _ = mg(x1, x2, 'WHU', return_aux=True)
train_md.BCEDiceLoss(mask_g, target).backward()
check(all(q.grad is None for q in mg.recon.parameters()),
      'the change loss sends no gradient into the reconstruction decoder (both computed in one pass)')
mg.zero_grad(set_to_none=True)

per_r = sum(mg.part_sizes().values())
shared_r = sum(q.numel() for q in mg.shared_parameters())
check(len(mg.recon_bn) == 5 and mg.part_sizes()['recon_bn'] == RECON_BN and per_r == 74160 + RECON_BN
      and count(mg) == count(md) + RECON_SHARED + len(DS) * RECON_BN and shared_r + len(DS) * per_r == count(mg),
      'every part on + reconstruction, 3 datasets: 5 reconstruction BatchNorms per dataset; '
      'total {:,} = shared {:,} + 3 x {:,}'.format(count(mg), shared_r, per_r))


def rec_step(model, opt, dataset):
    model.train()
    mask, aux = model(x1, x2, dataset, return_aux=True)
    loss = train_md.BCEDiceLoss(mask, target) + 0.1 * train_md.recon_mse(aux['recon'], x1, x2)
    if aux['heads']:
        loss = loss + torch.stack([train_md.BCEDiceLoss(h, target) for h in aux['heads'].values()]).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


m = copy.deepcopy(mg)
opt = torch.optim.Adam(m.parameters(), 5e-4, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
rec_step(m, opt, 'LEVIR')
rec_step(m, opt, 'SYSU')
before = {k: v.clone() for k, v in m.state_dict().items()}
rec_step(m, opt, 'WHU')
after = m.state_dict()
moved = [k for k in before if ('.per_dataset.LEVIR.' in k or '.per_dataset.SYSU.' in k)
         and not torch.equal(before[k], after[k])]
recon_moved = any(not torch.equal(before[k], after[k]) for k in after if k.startswith('recon.ups.'))
check(not moved and recon_moved, 'with reconstruction: a WHU step changes no LEVIR/SYSU tensor (%d changed) '
      'and trains the shared reconstruction convs' % len(moved))
m.eval()
with torch.no_grad():
    worst = max(max_diff(m(x1, x2, d), m(x2, x1, d)) for d in DS)
check(worst <= 1e-6, 'with reconstruction, trained: same change map with the dates swapped '
      '(max |difference| %.1e)' % worst)

img6 = np.random.RandomState(1).randint(0, 256, (8, 8, 6)).astype(np.uint8)
t6, _ = train_md.myTransforms.Compose([train_md.myTransforms.Normalize(mean=train_md.MEAN, std=train_md.STD),
                                       train_md.myTransforms.ToTensor()])(img6, np.zeros((8, 8), np.uint8))
want = torch.from_numpy(img6[:, :, ::-1].astype(np.float32).transpose(2, 0, 1).copy() / 255.0)
got = torch.cat([train_md.unnormalize(t6[None, 0:3].float()), train_md.unnormalize(t6[None, 3:6].float())], 1)[0]
check(max_diff(got, want) <= 1e-5, "PSNR helper: unnormalising the inputs gives back the original [0, 1] RGB "
      "pixels (max |difference| %.1e)" % max_diff(got, want))

print('\n=== 8. fusion (--fusion-stages, needs --recon) ===')
BLOCK = {1: 'cat4', 2: 'cat3', 3: 'cat2'}
WIDEN = {1: 256 * 96 * 9, 2: 128 * 32 * 9, 3: 64 * 48 * 9}   # new first-conv weights in cat4 / cat3 / cat2
HEAD = {2: 128 * 64 * 4 + 64 * 32 * 4 + 32 * 16 * 4 + 2 * (64 + 32 + 16) + 16 + 1}   # stage-2 head: 43,249
RECON_TO_2 = 320 * 256 * 4 + 2 * 256 + 256 * 128 * 4 + 2 * 128   # reconstruction stages 1-2: 459,520
torch.manual_seed(3)
mr3 = MultiDatasetNet(['LEVIR'], recon=True, base=copy.deepcopy(base))
rng_r = torch.get_rng_state()
torch.manual_seed(3)
mf = MultiDatasetNet(['LEVIR'], recon=True, fusion_stages=(2,), base=copy.deepcopy(base))
rng_f = torch.get_rng_state()
check(torch.equal(rng_r, rng_f), 'fusion on or off: same random stream afterwards (same batches)')
sr, sf = mr3.state_dict(), mf.state_dict()
wkey = 'swa.cat3.double_conv.0.weight'
check(all(torch.equal(sr[k], sf[k]) for k in sr if k != wkey)
      and all(k.startswith('fusion_heads.') for k in sf if k not in sr),
      'fusion on or off: every other weight (reconstruction decoder included) starts the same; '
      'the only new tensors are fusion_heads.*')
check(tuple(sf[wkey].shape) == (32, 192, 3, 3) and torch.equal(sf[wkey][:, :64], sr[wkey])
      and sf[wkey][:, 64:].abs().sum().item() == 0,
      "stage 2: cat3's first conv takes 64 + 128 inputs; the original 64 keep their weights, the new 128 start at zero")
for mode in ('eval', 'train'):
    d = same_output(mf, base, 'LEVIR', mode)
    check(d <= 1e-5, "%s mode, untrained: the change map is still BaseNet's (max |difference| %.1e)" % (mode, d))
n_change = base_total + WIDEN[2] + RECON_TO_2
n_only = sum(q.numel() for q in mf.training_only_parameters())
check(mf.fusion_widened[2] == WIDEN[2] and mf.fusion_head_parameter_count() == HEAD[2]
      and count(mf) == base_total + 503622 + WIDEN[2] + HEAD[2] and count(mf) - n_only == n_change,
      'stage 2: +{:,} change-decoder weights, +{:,} head; the change map uses {:,} of the {:,} '
      '(reconstruction stages 1-2 included)'.format(WIDEN[2], HEAD[2], n_change, count(mf)))
mf.eval()
with torch.no_grad():
    plain_f = mf(x1, x2, 'LEVIR')
    mask_f, aux_f = mf(x1, x2, 'LEVIR', return_aux=True)
check(torch.equal(plain_f, mask_f) and list(aux_f['heads']) == [2]
      and tuple(aux_f['heads'][2].shape) == (2, 1, 64, 64),
      'the change map is the same with or without the extra outputs; one stage-2 head map at full size')

mf.train()
mf.zero_grad(set_to_none=True)
train_md.BCEDiceLoss(mf(x1, x2, 'LEVIR'), target).backward()
gw = mf.swa.cat3.double_conv[0].weight.grad
late = list(mf.recon.ups)[2:] + list(mf.recon.norms)[2:] + [mf.recon.head_t1, mf.recon.head_t2, mf.fusion_heads]
check(gw is not None and gw[:, 64:].abs().sum().item() > 0
      and all(q.grad is None for mod in late for q in mod.parameters()),
      'change loss: the new cat3 weights get gradient (so fusion can switch on); reconstruction stages 3-5, '
      'the RGB outputs and the fusion head take no part')
opt_f = torch.optim.Adam(mf.parameters(), 5e-4, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
opt_f.step()
mf.zero_grad(set_to_none=True)
train_md.BCEDiceLoss(mf(x1, x2, 'LEVIR'), target).backward()
check(has_grad(mf.recon.ups[0]) and has_grad(mf.recon.ups[1]),
      'after one step: the change loss reaches reconstruction stages 1-2 through the fusion')
mf.zero_grad(set_to_none=True)
train_md.BCEDiceLoss(mf(x1, x2, 'LEVIR', return_aux=True)[1]['heads'][2], target).backward()
check(has_grad(mf.fusion_heads) and has_grad(mf.recon.ups[1]) and has_grad(mf.backbone.features[17])
      and all(q.grad is None for q in mf.swa.parameters())
      and all(q.grad is None for mod in list(mf.recon.ups)[2:] for q in mod.parameters()),
      'fusion-head loss: trains the head, reconstruction stages 1-2 and the encoder; never the change decoder '
      'or reconstruction stages 3-5')
mf.zero_grad(set_to_none=True)
mf.eval()
with torch.no_grad():
    sw = max_diff(mf(x1, x2, 'LEVIR'), mf(x2, x1, 'LEVIR'))
check(sw <= 1e-6, 'with fusion, after a step: same change map with the dates swapped (max |difference| %.1e)' % sw)

torch.manual_seed(4)
m123 = MultiDatasetNet(DS, recon=True, fusion_stages=(1, 2, 3), base=copy.deepcopy(base), **ALL_ON)
shapes = {k: tuple(getattr(m123.swa, BLOCK[k]).double_conv[0].weight.shape) for k in (1, 2, 3)}
check(shapes == {1: (96, 448, 3, 3), 2: (32, 192, 3, 3), 3: (48, 136, 3, 3)},
      'stages 1, 2, 3 with every part on: cat4 / cat3 / cat2 take 192+256 / 64+128 / 72+64 inputs')
for mode in ('eval', 'train'):
    worst = max(same_output(m123, base, d, mode) for d in DS)
    check(worst <= 1e-5, "stages 1-3, every part on, untrained, %s mode: BaseNet's change map for all 3 datasets "
          "(max |difference| %.1e)" % (mode, worst))
per_f = sum(m123.part_sizes().values())
shared_f = sum(q.numel() for q in m123.shared_parameters())
check(m123.part_sizes()['fusion_bn'] == 800 and shared_f + len(DS) * per_f == count(m123),
      'stages 1-3: the three heads\' BatchNorms are per dataset (800 each); total {:,} = shared {:,} + 3 x {:,}'
      .format(count(m123), shared_f, per_f))
opt = torch.optim.Adam(m123.parameters(), 5e-4, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
rec_step(m123, opt, 'LEVIR')
rec_step(m123, opt, 'SYSU')
before = {k: v.clone() for k, v in m123.state_dict().items()}
rec_step(m123, opt, 'WHU')
after = m123.state_dict()
moved = [k for k in before if ('.per_dataset.LEVIR.' in k or '.per_dataset.SYSU.' in k)
         and not torch.equal(before[k], after[k])]
check(not moved, 'stages 1-3, every part on: a WHU step changes no LEVIR/SYSU tensor (%d changed)' % len(moved))
try:
    MultiDatasetNet(['LEVIR'], fusion_stages=(2,), base=copy.deepcopy(base))
    refused_m = False
except ValueError:
    refused_m = True
check(refused_m, 'fusion without the reconstruction decoder is refused')

print('\n=== 9. tools/train_md.py end to end (synthetic data, CPU) ===')
tmp = tempfile.mkdtemp(prefix='check_md_')
rng = np.random.RandomState(0)


def make_dataset(root, n_train):
    for split, n in (('train', n_train), ('val', 4), ('test', 4)):
        for sub in ('A', 'B', 'label', 'list'):
            os.makedirs(os.path.join(root, split, sub))
        names = []
        for i in range(n):
            name = '%s_%d.png' % (split, i)
            for sub in ('A', 'B'):
                cv2.imwrite(os.path.join(root, split, sub, name),
                            rng.randint(0, 256, (64, 64, 3)).astype(np.uint8))
            label = np.zeros((64, 64), np.uint8)
            y, x = rng.randint(0, 40, 2)
            label[y:y + 20, x:x + 20] = 255
            cv2.imwrite(os.path.join(root, split, 'label', name), label)
            names.append(name)
        with open(os.path.join(root, split, 'list', split + '.txt'), 'w') as f:
            f.write('\n'.join(names))


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_md.main(argv)
    return buf.getvalue()


try:
    make_dataset(os.path.join(tmp, 'LEVIR'), 12)   # 6 batches of 2
    make_dataset(os.path.join(tmp, 'WHU'), 8)      # 4 batches of 2
    common = ['--data_root', tmp, '--inWidth', '64', '--inHeight', '64', '--batch_size', '2',
              '--num_workers', '0', '--device', 'cpu', '--max_steps', '8',
              '--savedir', os.path.join(tmp, 'results_md')]
    two = ['--datasets', 'LEVIR', 'WHU', '--domain-bn', '--enc-adapters', '--diff-adapters',
           '--dec-adapters'] + common
    log = run(two)
    check('Schedule: 4 rounds/epoch x 2 epochs = 8 rounds = 16 optimizer steps (8 per dataset)' in log,
          'two datasets: epoch = pass over the smaller set (4 rounds), 8 rounds = 8 steps per dataset')
    check('Per-dataset parameters: 74,160' in log, 'header: 74,160 parameters per dataset')
    check('Checkpoint selection: best mean val F1 over LEVIR, WHU' in log, 'header: selection on mean val F1')
    rows_on = re.findall(r'^  \[ON\] ', log, re.M)
    check('EXPERIMENT: LEVIR-WHU | bn-enc-diff-dec' in log and len(rows_on) == 4
          and '[off]' not in log, 'header: experiment name, all 4 parts shown [ON]')
    check(all(re.search(r'\[%s\] Test :\t Kappa \(te\) = ' % d, log) for d in ('LEVIR', 'WHU'))
          and 'Mean test F1 over LEVIR, WHU:' in log, 'a Test line per dataset, plus the mean')
    rdir = os.path.join(tmp, 'results_md_LEVIR-WHU_bn-enc-diff-dec_iter_8_lr_0.0005')
    check(all(os.path.isfile(os.path.join(rdir, f)) for f in
              ('best_model.pth', 'checkpoint.pth.tar', 'trainValLog.txt')),
          'run folder has best_model.pth, checkpoint.pth.tar, trainValLog.txt')
    with open(os.path.join(rdir, 'trainValLog.txt')) as f:
        check('EXPERIMENT: LEVIR-WHU | bn-enc-diff-dec' in f.read(), 'trainValLog.txt starts with the experiment header')
    try:
        run(two)
        refused = False
    except SystemExit:
        refused = True
    check(refused, 'the same command again (no --resume) refuses to overwrite the run')
    log_r = run(two + ['--resume'])
    check('=> resumed from' in log_r and '[WHU] Test :' in log_r, '--resume reloads the run and tests it')
    log1 = run(['--datasets', 'LEVIR'] + common)
    check('Schedule: 6 rounds/epoch x 2 epochs = 12 rounds = 12 optimizer steps (12 per dataset)' in log1,
          'one dataset: epochs, rounds and steps follow tools/train.py')
    tot = re.search(r'Total network parameters \(excluding idr\): (\d+)', log1)
    check(tot is not None and int(tot.group(1)) == base_total,
          'one dataset, every part off: %s parameters = BaseNet' % (tot.group(1) if tot else '?'))
    rows_off = re.findall(r'^  \[off\] ', log1, re.M)
    check('EXPERIMENT: LEVIR | shared' in log1 and len(rows_off) == 4 and '[ON]' not in log1,
          'one dataset, every part off: header shows all 4 parts [off]')
    check('Reconstruction task: off' in log1 and 'Recon(tr)' not in log1 and 'PSNR' not in log1,
          'without --recon: header says off; epoch and test lines as before')
    rec = ['--datasets', 'LEVIR', '--recon'] + common
    logr = run(rec)
    check('EXPERIMENT: LEVIR | shared | recon 0.1' in logr and 'Reconstruction task: ON' in logr
          and os.path.isdir(os.path.join(tmp, 'results_md_LEVIR_shared_rec0.1_iter_8_lr_0.0005')),
          'with --recon: header says ON (weight 0.1); run folder results_md_LEVIR_shared_rec0.1_...')
    check('Total network parameters (excluding idr): %d' % (base_total + 503622) in logr
          and 'Parameters the change map uses (training-only parts left out): {:,}'.format(base_total)
          in logr, 'with --recon: {:,} parameters in training, {:,} used for the change map'.format(
              base_total + 503622, base_total))
    check(re.search(r'Recon\(tr\) = \d+\.\d{4}\t PSNR\(val\) = -?\d+\.\d{2} dB', logr) is not None
          and re.search(r'\[LEVIR\] Test :\t Kappa \(te\) = ', logr) is not None
          and '[LEVIR] Test reconstruction: PSNR = ' in logr,
          'with --recon: epoch lines add Recon(tr) and PSNR(val); test adds a PSNR line')
    log_rr = run(rec + ['--resume'])
    check('=> resumed from' in log_rr and '[LEVIR] Test reconstruction: PSNR = ' in log_rr,
          'with --recon: --resume reloads the run and tests it')
    check('Fusion: off' in logr, 'with --recon only: header says fusion is off')
    fus = ['--datasets', 'LEVIR', '--recon', '--recon-weight', '0.5', '--fusion-stages', '2'] + common
    logf = run(fus)
    check('EXPERIMENT: LEVIR | shared | recon 0.5 | fusion stage 2 (head weight 1)' in logf
          and 'Fusion: ON' in logf
          and os.path.isdir(os.path.join(tmp, 'results_md_LEVIR_shared_rec0.5_fus2_iter_8_lr_0.0005')),
          'with --fusion-stages 2: header says so; run folder results_md_LEVIR_shared_rec0.5_fus2_...')
    check('Total network parameters (excluding idr): %d' % count(mf) in logf
          and 'Parameters the change map uses (training-only parts left out): {:,}'.format(n_change) in logf,
          'with fusion at stage 2: {:,} parameters in training, {:,} used for the change map'.format(
              count(mf), n_change))
    check(re.search(r'Recon\(tr\) = \d+\.\d{4}\t PSNR\(val\) = -?\d+\.\d{2} dB\t Head\(tr\) = \d+\.\d{4}', logf)
          is not None and '[LEVIR] Test reconstruction: PSNR = ' in logf,
          'with fusion: epoch lines add Head(tr) after Recon(tr) and PSNR(val); test as with --recon')
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            run(['--datasets', 'LEVIR', '--fusion-stages', '2'] + common)
            refused_f = False
        except SystemExit:
            refused_f = True
    check(refused_f, '--fusion-stages without --recon is refused')
    print('\n  Start of a run (two datasets, every part on; one dataset, every part off; one dataset with '
          '--recon; one with --recon and fusion at stage 2), and its epoch and test lines:')
    for line in (log + log1 + logr + logf).splitlines():
        if re.search(r'^=+$|EXPERIMENT|^Datasets|Per-dataset|part  |\[ON\]|\[off\]|Shared by|Adapter:|'
                     r'Reconstruction task|  decoder:|  loss:|  parameters:|Fusion:|  new input|'
                     r'  fusion change head|change map uses|'
                     r'Parameters:|Total network|Training:|Schedule|Checkpoint selection|Recon\(tr\)|'
                     r'Test :|Test reconstruction|Mean test', line):
            print('  | ' + line.rstrip())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
