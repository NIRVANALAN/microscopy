import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import yaml
import argparse
from easydict import EasyDict
from unet.model.ResNetUNet import ResNetUNet
from unet.train import train_model
from unet import loss


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 7
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    # Comment out to finetune further
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=10, gamma=0.1)
    num_epochs = args.get('epochs', 60)
    model = train_model(model, optimizer_ft, exp_lr_scheduler,
                        device=device, num_epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax classification loss")

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
