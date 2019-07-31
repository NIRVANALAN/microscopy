import torch
from torchvision import transforms
from .microscopy_dataset import MicroscopyClassification, MicroscopyKeypoint
from .samplers import DistributedGivenIterationSampler, DistributedTestSampler
from .collate import default_collate


def build_train_loader(args):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	if args.use_kpt:
		train_transform = transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])
		rotate = args.data.aug.get('rotate', 0)
		print(f'rotate: {rotate}')
		v_flip = args.data.aug.get('v_flip', False)
		h_flip = args.data.aug.get('h_flip', False)
		print(f'v_flip: {v_flip}, h_flipL{h_flip}')
		train_dataset = MicroscopyKeypoint(args.data.root, args.data.train_list,
										   args.data.train_bboxes_list, v_flip, h_flip, args.data.train_img_size,
										   args.data.test_img_size, args.data.scale, rotate=0, transform=train_transform,
										   args=args, train=True)
	else:
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		train_dataset = MicroscopyClassification(args.data.root,args.data.train_list,
												 args.data.train_img_size, train_transform)
	train_batch_size = args.data.train_batch_size
	if args.distributed:
		train_batch_size = train_batch_size // args.num_gpus
		train_sampler = DistributedGivenIterationSampler(
			train_dataset, total_iter=args.train.total_iter,
			batch_size=train_batch_size,
			world_size=args.num_gpus, rank=args.local_rank, last_iter=args.last_iter)
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=train_batch_size, shuffle=not args.distributed,
											   sampler=train_sampler if args.distributed else None,
											   collate_fn=default_collate, num_workers=args.data.workers,
											   pin_memory=True)
	return train_loader


def build_val_loader(args):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_transform = transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])
	if args.use_kpt:
		val_dataset = MicroscopyKeypoint( args.data.root,args.data.test_list,
										 args.data.test_bboxes_list, False, args.data.test_img_size, -1,
										 args.data.scale, 0, val_transform, args=args)
	else:
		val_dataset = MicroscopyClassification(args.data.root,args.data.test_list,
											   args.data.test_img_size, val_transform)
	test_batch_size = args.data.test_batch_size
	if args.distributed:
		test_batch_size = test_batch_size // args.num_gpus
		val_sampler = DistributedTestSampler(val_dataset, args.num_gpus, args.local_rank)
	val_loader = torch.utils.data.DataLoader(val_dataset,
											 sampler=val_sampler if args.distributed else None,
											 batch_size=test_batch_size, shuffle=False,
											 num_workers=args.data.workers, pin_memory=True,
											 collate_fn=default_collate, drop_last=False)
	return val_loader
