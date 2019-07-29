import os
import time
import yaml
import torch
import argparse
import random
import numpy as np
from prettytable import PrettyTable
from easydict import EasyDict
from tqdm import tqdm
from micoscopy.data import build_val_loader
from micoscopy.models import build_model
from micoscopy.util import AverageMeter, AveragePrecisionMeter
from micoscopy.dist import synchronize,get_rank


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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

    val_loader = build_val_loader(args)
    model = build_model(args)
    state_dict = torch.load(args.checkpoint)['state_dict']
    state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    device = torch.device('cuda')
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    test(args, model, val_loader, device)

def test(args, model, val_loader, device):
    model.eval()
    ap_meter = AveragePrecisionMeter()
    tp = torch.zeros(len(args.threshold), args.model.num_classes).to(device)
    pos_label = torch.zeros(len(args.threshold), args.model.num_classes).to(device)
    pos_pred = torch.zeros(len(args.threshold), args.model.num_classes).to(device)

    for batch_index, (data, labels) in enumerate(val_loader):
        if args.local_rank == 0 and batch_index % args.print_freq == 0:
            print ('{} [{}/{}]'.format(get_time(), batch_index, len(val_loader)))
        data, names = data
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        if type(output) == tuple:
            output = sum(output) / 3.0

        all_labels = [torch.zeros(args.data.test_batch_size, labels.size(1)).to(device) for _ in range(args.num_gpus)]
        all_output = [torch.zeros(args.data.test_batch_size, output.size(1)).to(device) for _ in range(args.num_gpus)]
        all_flags = [torch.zeros(args.data.test_batch_size).to(device) for _ in range(args.num_gpus)]
        real_batch_size = labels.size(0)
        tmp_labels = torch.zeros(args.data.test_batch_size, labels.size(1)).to(device)
        tmp_labels[:real_batch_size] += labels.data
        tmp_output = torch.zeros(args.data.test_batch_size, output.size(1)).to(device)
        tmp_output[:real_batch_size] += output.data
        tmp_flags = torch.zeros(args.data.test_batch_size).to(device)
        tmp_flags[:real_batch_size] += 1
        torch.distributed.all_gather(all_labels, tmp_labels)
        torch.distributed.all_gather(all_output, tmp_output)
        torch.distributed.all_gather(all_flags, tmp_flags)
        if args.local_rank == 0:
            all_flags = torch.stack(all_flags).view(-1).byte()
            all_labels = torch.stack(all_labels).view(-1, labels.size(1))[all_flags]
            all_output = torch.stack(all_output).view(-1, output.size(1))[all_flags]
            ap_meter.add(all_output.cpu(), all_labels.cpu())
        output = torch.sigmoid(output)
        output = output.data
        labels = labels.data
        for i in range(len(args.threshold)):
            pred = (output > args.threshold[i]).int()
            # pred [batch * num_classes], labels [batch * num_classes]
            tp[i] += torch.sum((pred==labels.int()).float() * labels, dim=0)
            pos_pred[i] += torch.sum(pred, dim=0).float()
            pos_label[i] += torch.sum(labels, dim=0).float()

    torch.distributed.all_reduce(tp)
    torch.distributed.all_reduce(pos_pred)
    torch.distributed.all_reduce(pos_label)
    precision = tp / pos_pred * 100.0
    recall = tp / pos_label * 100.0
    f1_score = 2.0 * tp / (pos_pred + pos_label) * 100.0
    if args.local_rank == 0:
        table = PrettyTable(['Metirc','Gun','Knife','Wrench','Pliers','Scissors','Mean'])
        row = ['Average Precision']
        row.extend(['{:.2f}'.format(100.0 * ap_meter.value()[i]) for i in range(5)])
        row.append('{:.2f}'.format(100.0 * ap_meter.value().mean()))
        table.add_row(row)
        for i in range(len(args.threshold)):
            row = ['P,R,F1 @ {:.2f}'.format(args.threshold[i])]
            row.extend(['{:.2f}, {:.2f}, {:.2f}'.format(precision[i][j], recall[i][j], f1_score[i][j])
                for j in range(5)])
            row.append('{:.2f}, {:.2f}, {:.2f}'.format(precision[i].mean(), recall[i].mean(), f1_score[i].mean()))
            table.add_row(row)
        print(table)
    return

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--threshold', nargs='*', default=[0.001, 0.01, 0.1, 0.5], type=float)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.checkpoint = args.checkpoint
    params.threshold = args.threshold
    params.local_rank = args.local_rank
    main(params)
