import os
import time
import yaml
import torch
import argparse
import random
import numpy as np
from easydict import EasyDict
from torch.nn.modules import Module
from microscopy.data import build_train_loader
from microscopy.models import build_model
from microscopy.util import AverageMeter, AveragePrecisionMeter, save_state
from microscopy.dist import synchronize
import pdb

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())




def chr_binary_cross_entropy_with_logits(input, target, pos_weight=None, weight=None, chr_weight=None, size_average=True, reduce=True):
    """
    Args:
            sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
            target: true value, one-hot-like vector of size [N,C]
            pos_weight: Weight for postive sample
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    # neg_abs = -input.abs()
    # loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        loss = input - input * target + log_weight * \
            (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    # if pos_weight is not None:
    #     # loss =  -(pos_weight * targets*torch.log(input+eps)+(1-targets)*torch.log(1-input+eps))

    #     # loss = (input).clamp(min=0) - input * \
    #     #     target + (1 + neg_abs.exp()).log()
    #     pos_weight = torch.where(target == 1, pos_weight, target)
    #     loss = loss*pos_weight

    if weight is not None:
        loss = loss * weight

    if chr_weight is not None:
        loss = loss * chr_weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class WeightedBCELogitsLoss(Module):
    def __init__(self, pos_weight=None, weight=None, WeightIsDynamic=False, size_average=True, reduce=True):
        """
        Args:
                pos_weight = Weight for postive samples. Size [1,C]
                weight = Weight for Each class. Size [1,C]
                PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
                WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        if isinstance(pos_weight, numbers.Number):
            self.register_buffer('pos_weight', torch.ones(pos_weight))
        else:
            self.register_buffer('pos_weight', pos_weight)  # tensor here
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target, chr_weight=None):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        # if self.PosWeightIsDynamic:
        #   positive_counts = target.sum(dim=0)
        #   nBatch = len(target)
        #   self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)
        if self.weight is None:
            return chr_binary_cross_entropy_with_logits(input, target,
                                                        pos_weight=self.pos_weight,
                                                        weight=None,
                                                        chr_weight=chr_weight,
                                                        size_average=self.size_average,
                                                        reduce=self.reduce)
        else:
            return chr_binary_cross_entropy_with_logits(input, target,
                                                        pos_weight=self.pos_weight, weight=self.weight, chr_weight=chr_weight, size_average=self.size_average, reduce=self.reduce)


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    print (f'Using distributed: {args.distributed}')
    if args.distributed:
        print (f'Local rank: {args.local_rank}')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = build_train_loader(args)

    model = build_model(args)
    device = torch.device('cuda')
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

    if args.loss.get('custom',False):
        print('using WeightedBCELogitsLoss')
        criterion = WeightedBCELogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    criterion.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.train.lr,
        momentum=args.train.momentum,
        weight_decay=args.train.weight_decay,
        nesterov=True
    )

    if args.resume:
        pass

    train(args, model, train_loader, criterion, optimizer, device)


def train(args, model, train_loader, criterion, optimizer, device):
    model.train()
    batch_times = AverageMeter(args.print_freq * 2)
    data_times = AverageMeter(args.print_freq * 2)
    losses = AverageMeter(args.print_freq * 2)
    end = time.time()
    for batch_index, (data, labels) in enumerate(train_loader):
        if batch_index in args.train.lr_iters:
            print('update learning rate')
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = param_group['lr'] * args.train.lr_gamma
        data_time_current = time.time() - end
        data_times.update(data_time_current)
        data, names = data
        data = data.to(device)
        labels = labels.to(device)
        #data = torch.autograd.Variable(data)
        #labels = torch.autograd.Variable(labels)
        output = model(data)
        if type(output) == tuple:
            loss = sum([criterion(o, labels) for o in output])
        else:
            loss = criterion(output, labels)
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
                  'Loss {:.4f} ({:.4f})'.format(
                  get_time(), batch_index, len(train_loader),
                  batch_time_current, batch_times.avg,
                  data_time_current, data_times.avg,
                  loss.data.item(), losses.avg
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
    parser=argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.local_rank = args.local_rank
    main(params)
