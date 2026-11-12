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
7. tools/train_md.py end to end on tiny synthetic data: two datasets with
   every part on, a refused re-run, --resume, and one dataset with every part
   off; the experiment header at the start says which parts are on and the
   parameter counts.
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

print('\n=== 7. tools/train_md.py end to end (synthetic data, CPU) ===')
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
    print('\n  Start of a run (two datasets, every part on; then one dataset, every part off), '
          'and its test lines:')
    for line in (log + log1).splitlines():
        if re.search(r'^=+$|EXPERIMENT|^Datasets|Per-dataset|part  |\[ON\]|\[off\]|Shared by|Adapter:|'
                     r'Parameters:|Total network|Training:|Schedule|Checkpoint selection|'
                     r'Test :|Mean test', line):
            print('  | ' + line.rstrip())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
