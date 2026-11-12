"""Train ONE Mobile-CDNet on several change-detection datasets at once, with
optional per-dataset parts (models/model_md.py).  Run from the repository root
on the DGX (MobileCDNet env), e.g.

    CUDA_VISIBLE_DEVICES=3 python tools/train_md.py --datasets LEVIR WHU SYSU \\
        --domain-bn --enc-adapters --diff-adapters --dec-adapters

Data: <data_root>/<DATASET>/{train,val,test}/{A,B,label,list} (the layout
tools/train.py reads), default data_root ~/mcd_data.

Everything that is not about multiple datasets is taken from tools/train.py:
BCE + Dice loss on the sigmoid output, Adam (betas 0.9/0.99, eps 1e-8, weight
decay 1e-4), poly learning rate with the same 200-step warm-up, seed 2333, the
same augmentations and normalisation, batch size 32, no validation after
epoch 0, and the pooled confusion-matrix scores of metric_tool.py.  (The
visualisation images train.py saves are not written.)

Multi-dataset schedule
  * Training goes in rounds: one batch from each dataset, in --datasets order,
    one optimizer step per batch.  A batch never mixes datasets.
  * --max_steps counts rounds, i.e. steps PER DATASET, so each dataset gets as
    many steps as in its own tools/train.py run.
  * One epoch = one pass over the smallest training set.  With a single
    dataset the epochs, learning-rate schedule and number of steps are exactly
    those of tools/train.py.
  * From epoch 1 on, every dataset's val split is scored after each epoch; the
    checkpoint with the best MEAN val F1 is kept and finally scored on every
    test split.
  * Gradients are cleared with zero_grad(set_to_none=True).  With plain
    zero_grad() (torch 1.8) the other datasets' parameters get zero gradients,
    and Adam keeps moving them through momentum and weight decay during this
    dataset's steps.
"""
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataset as myDataLoader  # noqa: E402
import Transforms as myTransforms  # noqa: E402
from metric_tool import ConfuseMatrixMeter  # noqa: E402
from models.model_md import MultiDatasetNet  # noqa: E402

DATASETS = ('LEVIR', 'WHU', 'SYSU')
SPLITS = ('train', 'val', 'test')
# Same as tools/train.py (OpenCV reads BGR; ToTensor flips the channel order).
MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
PART_FLAGS = (('domain_bn', 'bn'), ('enc_adapters', 'enc'),
              ('diff_adapters', 'diff'), ('dec_adapters', 'dec'))


def BCEDiceLoss(inputs, targets):
    """Same as tools/train.py: BCE + (1 - Dice) on the sigmoid output."""
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def adjust_learning_rate(args, optimizer, epoch, cur_round, total_rounds):
    """tools/train.py's schedule, counted in rounds (= steps with one dataset)."""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - cur_round * 1.0 / total_rounds) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and cur_round < 200:
        lr = args.lr * 0.9 * (cur_round + 1) / 200 + 0.1 * args.lr  # warm_up
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Cycle(object):
    """Endless batches from a DataLoader; a new (reshuffled) pass starts when one ends."""

    def __init__(self, loader):
        self.loader = loader
        self.it = None

    def next(self):
        if self.it is None:
            self.it = iter(self.loader)
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def to_device(img, target, device):
    pre = img[:, 0:3].to(device).float()
    post = img[:, 3:6].to(device).float()
    return pre, post, target.to(device).float()


def binarize(output):
    return torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()


@torch.no_grad()
def evaluate(model, loader, dataset, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for img, target in loader:
        pre, post, target = to_device(img, target, device)
        output = model(pre, post, dataset)
        losses.append(BCEDiceLoss(output, target).item())
        meter.update_cm(pr=binarize(output).cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / len(losses), meter.get_scores()


def train_epoch(args, model, optimizer, cycles, epoch, rounds_per_epoch, total_rounds, device):
    model.train()
    meters = {d: ConfuseMatrixMeter(n_class=2) for d in args.datasets}
    losses = {d: [] for d in args.datasets}
    lr = None
    for r in range(rounds_per_epoch):
        cur = epoch * rounds_per_epoch + r
        start = time.time()
        lr = adjust_learning_rate(args, optimizer, epoch, cur, total_rounds)
        for d in args.datasets:
            img, target = cycles[d].next()
            pre, post, target = to_device(img, target, device)
            output = model(pre, post, d)
            loss = BCEDiceLoss(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses[d].append(loss.item())
            with torch.no_grad():
                meters[d].update_cm(pr=binarize(output).cpu().numpy(), gt=target.cpu().numpy())
        if r % 5 == 0:
            left = (total_rounds - cur - 1) * (time.time() - start) / 3600
            last = ' '.join('%s %.3f' % (d, losses[d][-1]) for d in args.datasets)
            print('\rround: [%d/%d] lr: %.7f loss: %s time:%.3f h'
                  % (cur + 1, total_rounds, lr, last, left), end='')
    print()
    stats = {d: (sum(losses[d]) / len(losses[d]), meters[d].get_scores()) for d in args.datasets}
    return stats, lr


def run_dir(args):
    parts = [tag for flag, tag in PART_FLAGS if getattr(args, flag)]
    return '%s_%s_%s_iter_%d_lr_%s' % (args.savedir, '-'.join(args.datasets),
                                       '-'.join(parts) if parts else 'shared',
                                       args.max_steps, str(args.lr))


def parse_args(argv=None):
    p = ArgumentParser(description='Multi-dataset Mobile-CDNet with per-dataset parts')
    p.add_argument('--datasets', nargs='+', required=True, choices=DATASETS,
                   help='datasets trained together; each is <data_root>/<NAME>')
    p.add_argument('--data_root', default='~/mcd_data')
    p.add_argument('--inWidth', type=int, default=256)
    p.add_argument('--inHeight', type=int, default=256)
    p.add_argument('--max_steps', type=int, default=40000, help='steps PER DATASET (rounds)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--step_loss', type=int, default=100, help="epochs per x0.1 with --lr_mode step")
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr_mode', default='poly', help='poly or step')
    p.add_argument('--savedir', default='./results_md', help='prefix of the run folder')
    p.add_argument('--logFile', default='trainValLog.txt')
    p.add_argument('--seed', type=int, default=2333)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true',
                   help="continue the run in this run's folder from its last epoch")
    p.add_argument('--domain-bn', action='store_true', help='per-dataset BatchNorm (encoder + decoder)')
    p.add_argument('--enc-adapters', action='store_true', help='per-dataset adapters on the 5 encoder outputs')
    p.add_argument('--diff-adapters', action='store_true', help='per-dataset adapters on the 5 difference maps')
    p.add_argument('--dec-adapters', action='store_true', help='per-dataset adapters after the 12 decoder convs')
    p.add_argument('--adapter-reduction', type=int, default=16)
    p.add_argument('--adapter-dropout', type=float, default=0.1)
    args = p.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        p.error('--datasets lists a dataset twice')
    args.data_root = os.path.expanduser(args.data_root)
    return args


def main(argv=None):
    args = parse_args(argv)
    print('Called with args:')
    print(args)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available (use --device cpu only for tests)')

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = MultiDatasetNet(args.datasets, domain_bn=args.domain_bn,
                            enc_adapters=args.enc_adapters, diff_adapters=args.diff_adapters,
                            dec_adapters=args.dec_adapters, adapter_reduction=args.adapter_reduction,
                            adapter_dropout=args.adapter_dropout)

    savedir = run_dir(args)
    if os.path.isdir(savedir) and not args.resume:
        raise SystemExit('%s already exists. Add --resume to continue it, or move it away '
                         'to start again.' % savedir)
    roots = {}
    for d in args.datasets:
        roots[d] = os.path.join(args.data_root, d)
        for split in SPLITS:
            lst = os.path.join(roots[d], split, 'list', split + '.txt')
            if not os.path.isfile(lst):
                raise SystemExit('missing %s (expected <data_root>/%s/%s/{A,B,label,list})'
                                 % (lst, d, split))
    os.makedirs(savedir, exist_ok=True)
    model = model.to(device)

    total_params = sum(q.numel() for q in model.parameters())
    shared_params = sum(q.numel() for q in model.shared_parameters())
    sizes = model.part_sizes()
    per_dataset = sum(sizes.values())
    print('Total network parameters (excluding idr): ' + str(total_params))
    print('Datasets: %s  (data root %s)' % (', '.join(args.datasets), args.data_root))
    print('Per-dataset parts: ' + ' | '.join(
        '%s %s' % (flag, 'ON' if getattr(args, flag) else 'off') for flag, _ in PART_FLAGS))
    print('Parameters: total {:,} = shared {:,} + {} x {:,} per dataset'.format(
        total_params, shared_params, len(args.datasets), per_dataset))
    print('Per-dataset parameters: {:,} ({})'.format(
        per_dataset, ', '.join('{} {:,}'.format(k, v) for k, v in sizes.items())))

    train_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])
    eval_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    loaders = {}
    for d in args.datasets:
        sets = {s: myDataLoader.Dataset(s, file_root=roots[d], transform=train_tf if s == 'train' else eval_tf)
                for s in SPLITS}
        loaders[d] = {
            'train': torch.utils.data.DataLoader(
                sets['train'], batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=False, drop_last=True),
            'val': torch.utils.data.DataLoader(
                sets['val'], shuffle=False, batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False),
            'test': torch.utils.data.DataLoader(
                sets['test'], shuffle=False, batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False),
        }
        if len(loaders[d]['train']) == 0:
            raise SystemExit('%s: fewer training tiles than one batch' % d)
        print('  %-5s train %d tiles (%d batches per pass), val %d, test %d' % (
            d, len(sets['train']), len(loaders[d]['train']), len(sets['val']), len(sets['test'])))

    rounds_per_epoch = min(len(loaders[d]['train']) for d in args.datasets)
    max_epochs = int(np.ceil(args.max_steps / rounds_per_epoch))
    total_rounds = rounds_per_epoch * max_epochs
    print('Schedule: %d rounds/epoch x %d epochs = %d rounds = %d optimizer steps (%d per dataset)'
          % (rounds_per_epoch, max_epochs, total_rounds, total_rounds * len(args.datasets), total_rounds))
    print('Checkpoint selection: best mean val F1 over %s' % ', '.join(args.datasets))
    print('Run folder: %s' % savedir)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    config = {
        'datasets': list(args.datasets), 'domain_bn': args.domain_bn,
        'enc_adapters': args.enc_adapters, 'diff_adapters': args.diff_adapters,
        'dec_adapters': args.dec_adapters, 'adapter_reduction': args.adapter_reduction,
        'adapter_dropout': args.adapter_dropout, 'max_steps': args.max_steps,
        'batch_size': args.batch_size, 'lr': args.lr, 'lr_mode': args.lr_mode,
        'inWidth': args.inWidth, 'inHeight': args.inHeight, 'seed': args.seed,
    }
    ckpt_path = os.path.join(savedir, 'checkpoint.pth.tar')
    best_path = os.path.join(savedir, 'best_model.pth')
    start_epoch, best_mean, best_epoch = 0, 0.0, -1
    if args.resume:
        if not os.path.isfile(ckpt_path):
            raise SystemExit('nothing to resume: %s not found (no epoch after epoch 0 finished). '
                             'Move %s away and start again.' % (ckpt_path, savedir))
        ck = torch.load(ckpt_path, map_location=device)
        if ck['config'] != config:
            raise SystemExit('checkpoint settings differ from this command:\n  checkpoint %s\n  command    %s'
                             % (ck['config'], config))
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, best_mean, best_epoch = ck['epoch'], ck['best_mean'], ck['best_epoch']
        print('=> resumed from %s: next epoch %d, best mean val F1 %.4f (epoch %d). '
              'Batches restart in a new random order.' % (ckpt_path, start_epoch, best_mean, best_epoch))

    logger = open(os.path.join(savedir, args.logFile), 'a' if args.resume else 'w')
    if not args.resume:
        logger.write('Parameters: %d total, %d shared, %d per dataset\n'
                     % (total_params, shared_params, per_dataset))
        logger.write('Config: %s\n' % config)
        logger.write('Epoch\tDataset\tKappa (val)\tIoU (val)\tF1 (val)\tR (val)\tP (val)\n')
        logger.flush()

    cycles = {d: Cycle(loaders[d]['train']) for d in args.datasets}
    for epoch in range(start_epoch, max_epochs):
        stats, lr = train_epoch(args, model, optimizer, cycles, epoch, rounds_per_epoch,
                                total_rounds, device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if epoch == 0:
            continue  # tools/train.py does not validate after epoch 0

        val = {d: evaluate(model, loaders[d]['val'], d, device) for d in args.datasets}
        mean_f1 = float(np.mean([val[d][1]['F1'] for d in args.datasets]))
        for d in args.datasets:
            s = val[d][1]
            logger.write('%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'
                         % (epoch, d, s['Kappa'], s['IoU'], s['F1'], s['recall'], s['precision']))
        logger.write('%d\tmean F1\t\t\t%.4f\n' % (epoch, mean_f1))
        logger.flush()

        note = ''
        if best_mean <= mean_f1:
            best_mean, best_epoch = mean_f1, epoch
            torch.save(model.state_dict(), best_path)
            note = '  * new best, saved'
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_mean': best_mean,
                    'best_epoch': best_epoch, 'config': config, 'lr': lr}, ckpt_path)

        print('Epoch No. %d:\tmean F1(val) = %.4f\t(best %.4f @ epoch %d)%s'
              % (epoch, mean_f1, best_mean, best_epoch, note))
        for d in args.datasets:
            print('    %-5s\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f'
                  % (d, stats[d][0], val[d][0], stats[d][1]['F1'], val[d][1]['F1']))
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not os.path.isfile(best_path):
        raise SystemExit('no validated epoch, so no best model (need at least 2 epochs)')
    model.load_state_dict(torch.load(best_path, map_location=device))
    print('\nTesting the best checkpoint: epoch %d, mean val F1 %.4f' % (best_epoch, best_mean))
    test_f1 = []
    for d in args.datasets:
        _, s = evaluate(model, loaders[d]['test'], d, device)
        test_f1.append(s['F1'])
        print('[%s] Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f'
              % (d, s['Kappa'], s['IoU'], s['F1'], s['recall'], s['precision']))
        logger.write('Test\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'
                     % (d, s['Kappa'], s['IoU'], s['F1'], s['recall'], s['precision']))
    print('Mean test F1 over %s: %.4f' % (', '.join(args.datasets), float(np.mean(test_f1))))
    logger.write('Test\tmean F1\t\t\t%.4f\n' % float(np.mean(test_f1)))
    logger.close()


if __name__ == '__main__':
    main()
