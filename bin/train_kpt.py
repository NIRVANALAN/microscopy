import os
import time
import yaml
import torch
import argparse
import random
import numpy as np
from easydict import EasyDict
from microscopy.data import build_train_loader
from microscopy.models import build_model
from microscopy.util import AverageMeter, AveragePrecisionMeter, save_state, MSELoss, FocalLoss
from microscopy.dist import synchronize
import pdb


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    print(f'Using distributed: {args.distributed}')
    if args.distributed:
        print(f'Local rank: {args.local_rank}')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model = build_model(args)

    args.last_iter = -1
    if args.resume:
        checkpoint = torch.load(
            args.resume_path, map_location=lambda storage, loc: storage)
        state_dict = {k.replace('module.', ''): v for k,
                      v in checkpoint['state_dict'].items()}
        args.last_iter = checkpoint['iter']
        model.load_state_dict(state_dict)

    device = torch.device('cuda')
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.train.lr,
        momentum=args.train.momentum,
        weight_decay=args.train.weight_decay,
        nesterov=True
    )
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.loss.get('cls_loss', 'bce') == 'focal':
        gamma=args.loss.get('focal_gamma', 0)
        alpha=args.loss.get('focal_alpha',None)
        print(f'using focal loss with gamma {gamma} alpha {alpha}')
        criterion = FocalLoss(gamma=gamma, alpha=alpha)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(
            size_average=args.loss.cls_size_average, pos_weight=torch.Tensor(args.loss.pos_weight))
    criterion.to(device)

    if args.loss.get('kpt',None)=='mse':
        kpt_criterion = torch.nn.MSELoss(size_average=args.loss.kpt_size_average)
        print('kpt using MSELoss')
    else:
        kpt_criterion = torch.nn.BCEWithLogitsLoss(
            size_average=args.loss.kpt_size_average)
    kpt_criterion.to(device)

    train_loader = build_train_loader(args)
    torch.cuda.empty_cache()
    train(args, model, train_loader, criterion,
          kpt_criterion, optimizer, device)


def train(args, model, train_loader, criterion, kpt_criterion, optimizer, device):
    model.train()
    batch_times = AverageMeter(args.print_freq * 2)
    data_times = AverageMeter(args.print_freq * 2)
    cls_losses = AverageMeter(args.print_freq * 2)
    kpt_losses = AverageMeter(args.print_freq * 2)
    losses = AverageMeter(args.print_freq * 2)
    end = time.time()
    for batch_index, (data, labels, bboxes, kpt_labels) in enumerate(train_loader):
        batch_index += args.last_iter + 1
        if batch_index in args.train.lr_iters:
            print('update learning rate')
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = param_group['lr'] * args.train.lr_gamma
        data_time_current = time.time() - end
        data_times.update(data_time_current)
        data, names = data
        data = data.to(device)
        labels = labels.to(device)
        kpt_labels = kpt_labels.to(device)
        #data = torch.autograd.Variable(data)
        #labels = torch.autograd.Variable(labels)
        output = model(data)
        if len(output) == 2:
            output, kpt_output = output
            cls_loss = criterion(output, labels)
        elif len(output) == 4:
            kpt_output = output[-1]
            cls_loss = sum(map(lambda x: criterion(x, labels), output[:3]))
        # elif len(output) == 5:
        #    output,kpt1,kpt2,kpt3,kpt4 = output
        #    cls_loss = criterion(output, labels)
        #    kpt_output = kpt4
        kpt_loss = kpt_criterion(kpt_output, kpt_labels)
        loss = cls_loss + kpt_loss * args.loss.kpt_weight
        reduced_cls_loss = torch.Tensor([cls_loss.data.item()]).to(device)
        torch.distributed.all_reduce(reduced_cls_loss)
        cls_losses.update(reduced_cls_loss.data.item())
        reduced_kpt_loss = torch.Tensor([kpt_loss.data.item()]).to(device)
        torch.distributed.all_reduce(reduced_kpt_loss)
        kpt_losses.update(reduced_kpt_loss.data.item())
        reduced_loss = torch.Tensor([loss.data.item()]).to(device)
        torch.distributed.all_reduce(reduced_loss)
        losses.update(reduced_loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_current = time.time() - end
        batch_times.update(batch_time_current)
        if args.local_rank == 0 and batch_index % args.print_freq == 0:
            print('{} [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Loss {:.4f} ({:.4f})\t'
                  'Cls Loss {:.4f} ({:.4f})\t'
                  'Kpt Loss {:.4f} ({:.4f})'.format(
                      get_time(), batch_index, len(train_loader),
                      batch_time_current, batch_times.avg,
                      data_time_current, data_times.avg,
                      loss.data.item(), losses.avg,
                      cls_loss.data.item(), cls_losses.avg,
                      kpt_loss.data.item(), kpt_losses.avg
                  )
                  )
        end = time.time()
        if (batch_index + 1) % args.save_freq == 0 or batch_index == len(train_loader) - 1:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': batch_index + 1
            }
            if args.local_rank == 0:
                save_state(args.save_path, state, batch_index+1, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    params = EasyDict(config)
    params.seed = args.seed
    params.local_rank = args.local_rank
    main(params)
