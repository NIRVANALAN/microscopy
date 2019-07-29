import torch
from .chr_resnet import resnet50_chr, resnet101_chr
from .resnet import resnet50, resnet101
from .pose_resnet import resnet50_pose, resnet101_pose
from .chr_pose_resnet import resnet50_chr_pose, resnet101_chr_pose
from .fpn_resnet import resnet50_fpn, resnet101_fpn

MODELS = {
	'resnet50': resnet50,
	'resnet101': resnet101,
	'resnet50_chr': resnet50_chr,
	'resnet101_chr': resnet101_chr,
	'resnet50_pose': resnet50_pose,
	'resnet101_pose': resnet101_pose,
	'resnet50_chr_pose': resnet50_chr_pose,
	'resnet101_chr_pose': resnet101_chr_pose,
	'resnet50_fpn': resnet50_fpn,
	'resnet101_fpn': resnet101_fpn
}


def build_model(args):
	model = MODELS[args.model.arch](num_classes=args.model.num_classes)
	if args.model.pretrained:
		state_dict = torch.load(args.model.pretrained)
		state_dict = {k: v for k, v in state_dict.items() if not k in args.model.ignore}
		model.load_state_dict(state_dict, strict=False)
	return model
