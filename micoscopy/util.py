import math
import warnings
import torch
import numpy as np
import os
import time
from PIL import Image
import torch.nn.functional as F
from torch.nn.modules import Module
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm


# class _Loss(Module):
#     def __init__(self, size_average=None, reduce=None, reduction='mean'):
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction

def get_time():
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean', pos_weight=None):
	# type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
	r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
	Measures the element-wise mean squared error.
	See :class:`~torch.nn.MSELoss` for details.
	"""
	if not (target.size() == input.size()):
		warnings.warn("Using a target size ({}) that is different to the input size ({}). "
					  "This will likely lead to incorrect results due to broadcasting. "
					  "Please ensure they have the same size.".format(
			target.size(), input.size()),
			stacklevel=2)
	if size_average is not None or reduce is not None:
		reduction = _Reduction.legacy_get_string(size_average, reduce)
	if target.requires_grad:
		ret = (input - target) ** 2
		if pos_weight != None:
			pos_weight = torch.where(
				target == 1, pos_weight, torch.tensor(1.0))
			ret *= pos_weight
		if reduction != 'none':
			ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
	else:
		expanded_input, expanded_target = torch.broadcast_tensors(
			input, target)
		ret = torch._C._nn.mse_loss(
			expanded_input, expanded_target, _Reduction.get_enum(reduction))
	return ret


class MSELoss(Module):

	# __constants__ = ['reduction']

	def __init__(self, size_average=None, reduce=None, reduction='mean', pos_weight=None):
		super(MSELoss, self).__init__(size_average, reduce, reduction)
		self.pos_weight = pos_weight

	def forward(self, input, target):
		return mse_loss(input, target, reduction=self.reduction, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):

	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		print(f'Focal loss alphaL {self.alpha.data}')
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim() > 2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
		# target = target.view(-1, 1)

		# logpt = F.log_softmax(input, dim=1)
		# logpt = F.logsigmoid(input)
		pt = torch.sigmoid(input)
		pt = torch.where(target == 1, pt, 1 - pt)
		logpt = torch.log(pt)
		# logpt = logpt.gather(1,target)
		# logpt = logpt.view(-1)
		# pt = logpt.exp()

		if self.alpha is not None:
			if self.alpha.type() != input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			# at = self.alpha.gather(0, target.data)
			if self.alpha.dim != 2:
				at = torch.where(target == 1, self.alpha, 1 - target)  # like pos_weight
			else:
				at = torch.where(target == 1, self.alpha[0], self.alpha[1])
			logpt = logpt * at

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()


class AveragePrecisionMeter(object):
	"""
	The APMeter measures the average precision per class.
	The APMeter is designed to operate on `NxK` Tensors `output` and
	`target`, and optionally a `Nx1` Tensor weight where (1) the `output`
	contains model output scores for `N` examples and `K` classes that ought to
	be higher when the model is more convinced that the example should be
	positively labeled, and smaller when the model believes the example should
	be negatively labeled (for instance, the output of a sigmoid function); (2)
	the `target` contains only values 0 (for negative examples) and 1
	(for positive examples); and (3) the `weight` ( > 0) represents weight for
	each sample.
	"""

	def __init__(self, difficult_examples=True):
		super(AveragePrecisionMeter, self).__init__()
		self.reset()
		self.difficult_examples = difficult_examples

	def reset(self):
		"""Resets the meter with empty member variables"""
		self.scores = torch.FloatTensor(torch.FloatStorage())
		self.targets = torch.LongTensor(torch.LongStorage())

	def add(self, output, target):
		"""
		Args:
			output (Tensor): NxK tensor that for each of the N examples
				indicates the probability of the example belonging to each of
				the K classes, according to the model. The probabilities should
				sum to one over all classes
			target (Tensor): binary NxK tensort that encodes which of the K
				classes are associated with the N-th input
					(eg: a row [0, 1, 0, 1] indicates that the example is
						 associated with classes 2 and 4)
			weight (optional, Tensor): Nx1 tensor representing the weight for
				each example (each weight > 0)
		"""
		if not torch.is_tensor(output):
			output = torch.from_numpy(output)
		if not torch.is_tensor(target):
			target = torch.from_numpy(target)

		if output.dim() == 1:
			output = output.view(-1, 1)
		else:
			assert output.dim() == 2, \
				'wrong output size (should be 1D or 2D with one column \
				per class)'
		if target.dim() == 1:
			target = target.view(-1, 1)
		else:
			assert target.dim() == 2, \
				'wrong target size (should be 1D or 2D with one column \
				per class)'
		if self.scores.numel() > 0:
			assert target.size(1) == self.targets.size(1), \
				'dimensions for output should match previously added examples.'

		# make sure storage is of sufficient size
		if self.scores.storage().size() < self.scores.numel() + output.numel():
			new_size = math.ceil(self.scores.storage().size() * 1.5)
			self.scores.storage().resize_(int(new_size + output.numel()))
			self.targets.storage().resize_(int(new_size + output.numel()))

		# store scores and targets
		offset = self.scores.size(0) if self.scores.dim() > 0 else 0

		# print(offset + output.size(0), output.size(1))
		self.scores.resize_(offset + output.size(0), output.size(1))
		self.targets.resize_(offset + target.size(0), target.size(1))
		self.scores.narrow(0, offset, output.size(0)).copy_(output)
		self.targets.narrow(0, offset, target.size(0)).copy_(target)

	def value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""

		if self.scores.numel() == 0:
			return 0
		ap = torch.zeros(self.scores.size(1))
		rg = torch.arange(1, self.scores.size(0)).float()

		# compute average precision for each class
		for k in range(self.scores.size(1)):
			# sort scores
			scores = self.scores[:, k]
			targets = self.targets[:, k]

			# compute average precision
			ap[k] = AveragePrecisionMeter.average_precision(
				scores, targets, self.difficult_examples)
		return ap

	@staticmethod
	def average_precision(output, target, difficult_examples=True):

		# sort examples
		sorted, indices = torch.sort(output, dim=0, descending=True)

		# Computes prec@i
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		for i in indices:
			label = target[i]
			# if difficult_examples and label == 0:
			#    continue
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, length=0):
		self.length = length
		self.reset()

	def reset(self):
		if self.length > 0:
			self.history = []
		else:
			self.count = 0
			self.sum = 0.0
		self.val = 0.0
		self.avg = 0.0

	def update(self, val):
		if self.length > 0:
			self.history.append(val)
			if len(self.history) > self.length:
				del self.history[0]

			self.val = self.history[-1]
			self.avg = np.mean(self.history)
		else:
			self.val = val
			self.sum += val
			self.count += 1
			self.avg = self.sum / self.count


def save_state(path, state, iter, best):
	if not os.path.exists(path):
		os.makedirs(path)
	print('saving to {}/iter_{}.pth.tar'.format(path, iter))
	torch.save(state, '{}/iter_{}.pth.tar'.format(path, iter))
	if best:
		torch.save(state, '{}/best.pth.tar'.format(path))
