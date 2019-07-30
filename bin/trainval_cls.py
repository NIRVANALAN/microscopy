import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
from prettytable import PrettyTable
from easydict import EasyDict
from microscopy.data import build_train_loader, build_val_loader
from microscopy.models import build_model
from microscopy.util import AverageMeter, AveragePrecisionMeter, save_state, FocalLoss, get_time
from microscopy.dist import synchronize

if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())


def main(args):
	# num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
	# args.num_gpus = num_gpus
	args.distributed = False
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
		checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
		state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
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

	if args.loss.get('cls_loss', None) == 'focal':
		gamma = args.loss.get('focal_gamma', 0)
		alpha = args.loss.get('focal_alpha', None)
		print(f'using focal loss with gamma {gamma} alpha {alpha}')
		criterion = FocalLoss(gamma=gamma, alpha=alpha)
	elif args.loss.get('cls_loss', None) == 'CE':
		criterion = torch.nn.CrossEntropyLoss(size_average=args.loss.cls_size_average)

	criterion.to(device)

	train_loader = build_train_loader(args)
	torch.cuda.empty_cache()
	train(args, model, train_loader, criterion, optimizer, device)


def train(args, model, train_loader, criterion, optimizer, device):
	model.train()
	batch_times = AverageMeter(args.print_freq * 2)
	data_times = AverageMeter(args.print_freq * 2)
	cls_losses = AverageMeter(args.print_freq * 2)
	losses = AverageMeter(args.print_freq * 2)
	end = time.time()
	best_ap = 0.0
	for batch_index, (data, labels) in enumerate(train_loader):
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
		output = model(data)
		cls_loss = criterion(output, labels)
		loss = cls_loss
		reduced_cls_loss = torch.Tensor([cls_loss.data.item()]).to(device)
		# torch.distributed.all_reduce(reduced_cls_loss)
		cls_losses.update(reduced_cls_loss.data.item())
		reduced_loss = torch.Tensor([loss.data.item()]).to(device)
		# torch.distributed.all_reduce(reduced_loss)
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
				.format(
				get_time(), batch_index, len(train_loader),
				batch_time_current, batch_times.avg,
				data_time_current, data_times.avg,
				loss.data.item(), losses.avg,
				cls_loss.data.item(), cls_losses.avg,
			)
			)
		end = time.time()
		if (batch_index + 1) % args.save_freq == 0 or batch_index == len(train_loader) - 1:
			torch.cuda.empty_cache()
			ap = test(args, model, device)
			model.train()
			torch.cuda.empty_cache()
			state = {
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iter': batch_index + 1
			}
			if args.local_rank == 0:
				save_state(args.save_path, state, batch_index + 1, ap > best_ap)
				if ap > best_ap:
					best_ap = ap
				print('{}: Curr AP:{:.2f} Best AP:{:.2f}'.format(get_time(),
																 ap, best_ap))


def test(args, model, device):
	model.eval()
	ap_meter = AveragePrecisionMeter()
	args.threshold = args.get('threshold', [0.001, 0.01, 0.1, 0.5])
	tp = torch.zeros(len(args.threshold), args.model.num_classes).to(device)
	pos_label = torch.zeros(len(args.threshold), args.model.num_classes).to(device)
	pos_pred = torch.zeros(len(args.threshold), args.model.num_classes).to(device)
	true_bbox = torch.zeros(args.model.num_classes).to(device)
	num_bbox = torch.zeros(args.model.num_classes).to(device)

	val_loader = build_val_loader(args)
	with torch.no_grad():
		for batch_index, (data, labels, bboxes, kpt_labels) in enumerate(val_loader):
			if args.local_rank == 0 and batch_index % args.print_freq == 0:
				print('{} [{}/{}]'.format(get_time(), batch_index, len(val_loader)))
			data, names = data
			data = data.to(device).requires_grad_(False)
			labels = labels.to(device).requires_grad_(False)
			output = model(data)
			if len(output) == 2:
				output, kpt_output = output
			elif len(output) == 4:
				o1, o2, o3, kpt_output = output
				output = o3
			test_batch_size = args.data.test_batch_size // args.num_gpus
			all_labels = [torch.zeros(test_batch_size, labels.size(1)).to(device) for _ in range(args.num_gpus)]
			all_output = [torch.zeros(test_batch_size, output.size(1)).to(device) for _ in range(args.num_gpus)]
			all_flags = [torch.zeros(test_batch_size).to(device) for _ in range(args.num_gpus)]
			real_batch_size = labels.size(0)
			tmp_labels = torch.zeros(test_batch_size, labels.size(1)).to(device)
			tmp_labels[:real_batch_size] += labels.data
			tmp_output = torch.zeros(test_batch_size, output.size(1)).to(device)
			tmp_output[:real_batch_size] += output.data
			tmp_flags = torch.zeros(test_batch_size).to(device)
			tmp_flags[:real_batch_size] += 1
			# torch.distributed.all_gather(all_labels, tmp_labels)
			# torch.distributed.all_gather(all_output, tmp_output)
			# torch.distributed.all_gather(all_flags, tmp_flags)
			# if args.local_rank == 0:
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
				tp[i] += torch.sum((pred == labels.int()).float() * labels, dim=0)
				pos_pred[i] += torch.sum(pred, dim=0).float()
				pos_label[i] += torch.sum(labels, dim=0).float()
			# synchronize()
			w, h = kpt_output.size()[2:]
			kpt_output = kpt_output.view(kpt_output.size(0), kpt_output.size(1), -1)
			all_pred = torch.argmax(kpt_output, dim=2)
			for i in range(all_pred.size(0)):
				pred = [((all_pred[i][j].item() % w + 0.5) / w, (all_pred[i][j].item() / w + 0.5) / h) for j in
						range(5)]
				tmp_true_bbox = torch.zeros(args.model.num_classes).to(device)
				tmp_num_bbox = torch.zeros(args.model.num_classes).to(device)
				for bbox in bboxes[i]:
					cls = bbox[0]
					tmp_num_bbox[cls] = 1.0
					if bbox[1] < pred[cls][0] < bbox[3] and bbox[2] < pred[cls][1] < bbox[4]:
						tmp_true_bbox[cls] = 1.0
				true_bbox += tmp_true_bbox
				num_bbox += tmp_num_bbox

	# torch.distributed.all_reduce(tp)
	# torch.distributed.all_reduce(pos_pred)
	# torch.distributed.all_reduce(pos_label)
	# torch.distributed.all_reduce(true_bbox)
	# torch.distributed.all_reduce(num_bbox)
	precision = tp / pos_pred * 100.0
	recall = tp / pos_label * 100.0
	f1_score = 2.0 * tp / (pos_pred + pos_label) * 100.0
	localization = true_bbox / num_bbox * 100.0
	if args.local_rank == 0:
		table = PrettyTable(['T4', 'T4R'])
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
		row = ['Localization']
		row.extend(['{:.2f}'.format(localization[i]) for i in range(5)])
		row.append('{:.2f}'.format(localization.mean()))
		table.add_row(row)
		print(table)
		return 100.0 * ap_meter.value().mean()
	return 0.0


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
