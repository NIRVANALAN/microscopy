import os
import time
import yaml
import torch
import argparse
import random
import numpy as np
import cv2
from prettytable import PrettyTable
from easydict import EasyDict
from tqdm import tqdm
from microscopy.data import build_val_loader
from microscopy.models import build_model
from microscopy.util import AverageMeter, AveragePrecisionMeter
from microscopy.dist import synchronize,get_rank


object_categories = ['Gun','Knife','Wrench','Pliers','Scissors']

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
    state_dict = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)['state_dict']
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
    if args.local_rank == 0 and not os.path.exists(args.output):
        os.makedirs(args.output)
    test(args, model, val_loader, device)

def test(args, model, val_loader, device):
    model.eval()
    for batch_index, (data, labels, bboxes, kpt_labels) in enumerate(val_loader):
        if args.local_rank == 0 and batch_index % args.print_freq == 0:
            print('{} [{}/{}]'.format(get_time(), batch_index, len(val_loader)))
        data, names = data
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        if len(output) == 2:
            output, kpt_output = output
        elif len(output) == 4:
            o1, o2, o3, kpt_output = output
            output = o3
        kpt_output = torch.sigmoid(kpt_output)
        output = torch.sigmoid(output).data
        w,h = kpt_output.size()[2:]
        tmp_kpt_output = kpt_output.view(kpt_output.size(0), kpt_output.size(1), -1)
        pred = torch.argmax(tmp_kpt_output, dim=2)
        for i in range(pred.size(0)):
            img = cv2.imread(names[i])
            text = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
                output[i][0].item(), output[i][1].item(), output[i][2].item(),
                output[i][3].item(), output[i][4].item())
            cv2.putText(img, text, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
            ori_h, ori_w = img.shape[:2]
            true_pred = [((pred[i][j].item()%w+0.5)/w, (pred[i][j].item()/w+0.5)/h) for j in range(5)]
            for bbox in bboxes[i]:
                cls = bbox[0]
                cv2.rectangle(img, (int(bbox[1]*ori_w),int(bbox[2]*ori_h)),
                    (int(bbox[3]*ori_w), int(bbox[4]*ori_h)),(0,0,255),3)
                cv2.circle(img, (int(true_pred[cls][0]*ori_w),int(true_pred[cls][1]*ori_h)), 5, (0,0,255), -1)
            if len(bboxes[i]):
                cv2.imwrite(os.path.join(args.output, f'{batch_index}_{i}.jpg'), img)
                for j in range(5):
                    if labels[i][j]==1:
                        # mask = kpt_output[i][j].data.cpu().numpy().transpose()
                        mask = kpt_output[i][j].data.cpu().numpy()
                        mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask)) * 255.0
                        mask = np.ones((h,w,3)) * mask.reshape((h, w, 1))
                        mask = mask.astype(np.uint8)
                        cv2.imwrite(os.path.join(args.output, f'{batch_index}_{i}_mask{j}.jpg'), mask)
                        gt = kpt_labels[i][j].cpu().numpy().transpose()
                        gt = np.ones((h,w,3)) * gt.reshape((h,w,1)) * 255.0
                        gt = gt.astype(np.uint8)
                        cv2.imwrite(os.path.join(args.output, f'{batch_index}_{i}_gt{j}.jpg'), gt)
    return

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--output', type=str, default='visualize')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.checkpoint = args.checkpoint
    params.local_rank = args.local_rank
    params.output = args.output
    main(params)
