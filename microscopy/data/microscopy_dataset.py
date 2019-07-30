import traceback
import os
import io
import sys
import cv2
import scipy.stats as st
import PIL
import numpy as np
import torch
from torchvision.transforms.functional import vflip, hflip
import torch.utils.data as data
from PIL import Image, ImageDraw

object_categories = ['T4', 'T4R', 'S1']
masks = ['Cytoskeleton', 'Desmosome', 'LipidDroplet', 'MitochondriaDark', 'MitochondriaLight', 'NuclearMembrane',
		 'PlasmaMembrane']


def pil_2_cv2(img):
	return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def cv2_2_pil(img):
	return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def read_object_labels(file, header=True):
	images = []
	num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		for line in f:
			img, cell_type, mask_label = line.split(';')
			cell_label = object_categories.index(cell_type)
			mask_label = eval(mask_label.strip('\n'))
			mask_label = (np.asarray(mask_label)).astype(np.float32)
			mask_label = torch.from_numpy(mask_label)
			images.append((img, cell_label, mask_label))
	return images


def pil_loader(img_str):
	buff = io.BytesIO(img_str)
	with Image.open(buff) as img:
		img = img.convert('RGB')
	return np.array(img)


class MicroscopyClassification(data.Dataset):
	def __init__(self, train_list, img_size, transform=None, target_transform=None, crop_size=-1):
		# self.root = root
		self.img_size = img_size
		# self.path_images = os.path.join(root, 'JPEGImage')
		# self.path_annotation = os.path.join(root, 'Annotation')

		self.transform = transform
		self.crop_size = crop_size
		self.target_transform = target_transform

		self.classes = object_categories
		self.images = read_object_labels(train_list)

		print('[dataset] Microscopy classification number of classes=%d  number of images=%d' % (
			len(self.classes), len(self.images)))

	def __getitem__(self, index):
		path, target, mask_target = self.images[index]
		img = Image.open(path).convert('RGB')
		img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
		# if self.crop_size > 0:
		# 	start_w = int((self.img_size - self.crop_size) * np.random.random())
		# 	start_h = int((self.img_size - self.crop_size) * np.random.random())
		# 	img = img.crop((start_w, start_h, start_w +
		# 					self.crop_size, start_h + self.crop_size))

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return (img, path), target

	def __len__(self):
		return len(self.images)

	def get_number_classes(self):
		return len(self.classes)


def gkern(kernlen=21, nsig=3):
	"""Returns a 2D Gaussian kernel."""

	x = np.linspace(-nsig, nsig, kernlen + 1)
	kern1d = np.diff(st.norm.cdf(x))
	kern2d = np.outer(kern1d, kern1d)
	return kern2d / kern2d.max()


class MicroscopyKeypoint(data.Dataset):
	def __init__(self, root, file_csv, file_bboxes, vertical_flip, horizonal_flip, img_size, crop_size=-1,
				 scale=4, rotate=0, transform=None, target_transform=None, args=None, save_pos=False, train=False):
		self.root = root
		self.path_images = os.path.join(root, 'JPEGImage')
		self.path_annotation = os.path.join(root, 'Annotation')
		self.train = train
		self.args = args
		self.loc_weight = self.args.get('loc_weight', 1)
		print('loc weight:{}'.format(self.loc_weight))
		if self.loc_weight != 0:
			self.gaussian_len = args.get('gaussian_len', 20)
			print('gaussian kernel len: {}'.format(self.gaussian_len))
		self.save_pos = '/mnt/lustre/share/reid/SIXray/tmp' if save_pos else None
		if train and args.data.get('fake', False):
			self.crop_dir = args.data.crop_gt_dir
			self.aug_type = args.data.fake.type
			self.crop_prob = args.data.fake.p
			print('aug prob:{}'.format(self.crop_prob))
			self.crop_standard = [0, 0, [117, 102], [
				103, 101], [88, 84]]
			self.blend_alpha = args.data.fake.get('alpha', 0.5)
			print('alpha:{}'.format(self.blend_alpha))
			self.fake_label = args.data.fake.get('fake_label', 1)
			print(self.fake_label)
		# self.strategy = args.data.fake.get('strategy', 'all')
		else:
			self.aug_type = None
		self.transform = transform
		self.target_transform = target_transform
		self.img_size = img_size
		self.crop_size = crop_size
		self.h_flip = horizonal_flip
		self.v_flip = vertical_flip
		self.rotate = rotate
		self.scale = scale
		self.num_classes = len(object_categories)
		self.images = read_object_labels(file_csv, self.path_annotation)
		self.bboxes = [[] for _ in range(len(self.images))]
		img_to_idx = {image[0]: i for i, image in enumerate(self.images)}
		category_to_idx = {category: i for i,
										   category in enumerate(object_categories)}
		with open(file_bboxes, 'r') as f:
			for line in f:
				sp = line.strip().split()
				pos = list(map(float, sp[2:]))
				self.bboxes[img_to_idx[sp[0]]].append(
					[category_to_idx[sp[1]], pos[0], pos[1], pos[2], pos[3]])
		print('[dataset] X-ray classification number of classes=%d  number of images=%d' % (
			self.num_classes, len(self.images)))

	def __getitem__(self, index):
		path, target = self.images[index]
		bboxes = self.bboxes[index]
		img = Image.open(os.path.join(self.path_images,
									  f'{path}.jpg')).convert('RGB')
		width, height = img.size
		img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

		kpt = np.zeros((self.num_classes + 1, self.img_size //
						self.scale, self.img_size // self.scale))
		kpt[-1] = 1.0
		get_fake = False
		for bbox in bboxes:
			x1 = int((self.img_size - 1.0) * bbox[1] / width / self.scale)
			x2 = int((self.img_size - 1.0) * bbox[3] / width / self.scale) + 1
			y1 = int((self.img_size - 1.0) * bbox[2] / height / self.scale)
			y2 = int((self.img_size - 1.0) * bbox[4] / height / self.scale) + 1
			if self.args.get('kpt_type', None) == 'gaussian':
				kernel_len = min(min(x2 - x1, y2 - y1), self.args.get('gaussian_len', 20))
				gaussian_kernel = gkern(kernel_len) * self.loc_weight
				mid_x = (x1 + x2) / 2
				mid_y = (y1 + y2) / 2
				shape = kpt[bbox[0], int(mid_x - kernel_len / 2):int(mid_x + kernel_len / 2):,
						int(mid_y - kernel_len / 2):int(mid_y + kernel_len / 2)].shape
				# if kernel_len == x2 - x1:
				try:
					kpt[bbox[0], int(mid_x - kernel_len / 2):int(mid_x + kernel_len / 2):,
					int(mid_y - kernel_len / 2):int(mid_y + kernel_len / 2)] = gaussian_kernel[:shape[0], :shape[1]]
				except:
					print('bbox x1,x2,y1,y2:', x1, x2,
						  y1, y2, gaussian_kernel.shape)
					print('yrange:{},{}'.format(
						int((x2 - x1 - y2 + y1) / 2), int((x2 - x1 + y2 - y1) / 2)))
					print('kpt shape:{}'.format(shape))
					traceback.print_exc()
			# else:
			# try:
			#     kpt[bbox[0], x1:x2, y1:y2] = gaussian_kernel[int(
			#         (y2-y1-x2+x1)/2):int((y2-y1-x2+x1)/2)+shape[0], :shape[1]]
			# except:
			#     print('bbox y1,y2:', y1, y2, gaussian_kernel.shape)
			else:
				kpt[bbox[0], x1:x2, y1:y2] = 1.0 * self.loc_weight
			kpt[-1, x1:x2, y1:y2] = 0.0
		# w = int((self.img_size-1) * (bbox[1]+np.random.random()*(bbox[3]-bbox[1])) / width / self.scale)
		# h = int((self.img_size-1) * (bbox[2]+np.random.random()*(bbox[4]-bbox[2])) / height / self.scale)
		# kpt[bbox[0]][w][h] = 1.0
		# kpt[-1][w][h] = 0.0

		if self.aug_type == 'paste':
			if torch.max(target) != 1 and np.random.random() >= 1 - self.crop_prob:
				# for i in range(len(list_target)):
				fake_target_index = (1 - target).nonzero()
				i = fake_target_index[torch.randint(
					2, len(fake_target_index), (1,)).item()].item()
				get_fake = i
				target[i] = 1
				dst_dir = os.path.join(self.crop_dir, str(i))
				img_num = os.listdir(dst_dir).__len__()
				pos_img = Image.open(os.path.join(
					dst_dir, '{}.jpg'.format(np.random.randint(img_num))))
				# add resized crop fake234 image
				if not ((pos_img.size[0] * pos_img.size[1] < self.crop_standard[i][0] * self.crop_standard[i][
					1]) and all([a < b for a, b in zip(pos_img.size, self.crop_standard[i])])):
					scale_index = np.argmax(pos_img.size)
					crop_scale = self.crop_standard[i][scale_index] / \
								 pos_img.size[scale_index]
					pos_img = pos_img.resize(
						[int(crop_scale * s) for s in pos_img.size])
				x1 = np.random.randint(img.size[0] - pos_img.size[0])
				y1 = np.random.randint(img.size[1] - pos_img.size[1])
				img.paste(pos_img, (x1, y1))
				x2 = x1 + pos_img.size[0]
				y2 = y1 + pos_img.size[1]
				x1 = int(x1 / self.scale)
				y1 = int(y1 / self.scale)
				x2 = int(x2 / self.scale) + 1
				y2 = int(y2 / self.scale) + 1
				kpt[i, x1:x2, y1:y2] = 1.0
				kpt[-1, x1:x2, y1:y2] = 0.0

		if self.aug_type == 'blend':
			if torch.max(target) != 1 and np.random.random() >= 1 - self.crop_prob:
				# for i in range(len(list_target)):
				fake_target_index = (1 - target).nonzero()
				i = fake_target_index[torch.randint(
					2, len(fake_target_index), (1,)).item()].item()
				get_fake = i
				target[i] = self.fake_label
				dst_dir = os.path.join(self.crop_dir, str(i))
				img_num = os.listdir(dst_dir).__len__()
				pos_img = PIL.Image.open(os.path.join(
					dst_dir, '{}.jpg'.format(np.random.randint(img_num))))
				# print('origin pos_img size:{}'.format(pos_img.size))
				if not ((pos_img.size[0] * pos_img.size[1] < self.crop_standard[i][0] * self.crop_standard[i][
					1]) and all([a < b for a, b in zip(pos_img.size, self.crop_standard[i])])):
					scale_index = np.argmax(pos_img.size)
					crop_scale = self.crop_standard[i][scale_index] / \
								 pos_img.size[scale_index]
					pos_img = pos_img.resize(
						[int(crop_scale * s) for s in pos_img.size])
				# m   anipulate via cv2
				# print('image size:{},{}'.format(img.size,pos_img.size))
				cv_pos_img = pil_2_cv2(pos_img.copy())
				cv_img = pil_2_cv2(img.copy())
				offset_x, offset_y = np.random.randint(
					cv_img.shape[0] - cv_pos_img.shape[0]), np.random.randint(cv_img.shape[1] - cv_pos_img.shape[1])
				crop_shape = cv_pos_img.shape
				# print('crop_shape:{}'.format(crop_shape))
				crop_img = cv_img[offset_x: offset_x +
											crop_shape[0], offset_y:offset_y + crop_shape[1]]
				# print(crop_img.shape, cv_pos_img.shape)
				assert crop_img.shape == cv_pos_img.shape
				crop_img = cv2.addWeighted(
					crop_img, self.blend_alpha, cv_pos_img, 1 - self.blend_alpha, 0)
				cv_img[offset_x: offset_x + crop_shape[0],
				offset_y:offset_y + crop_shape[1]] = crop_img
				img = cv2_2_pil(cv_img)
				x1 = offset_x
				y1 = offset_y
				x2 = offset_x + crop_shape[0]
				y2 = offset_y + crop_shape[1]
				x1 = int(x1 / self.scale)
				y1 = int(y1 / self.scale)
				x2 = int(x2 / self.scale) + 1
				y2 = int(y2 / self.scale) + 1
				if self.args.get('kpt_type', None) == 'gaussian':
					kernel_len = min(min(x2 - x1, y2 - y1), self.args.get('gaussian_len', 20))
					gaussian_kernel = gkern(kernel_len) * self.loc_weight
					mid_x = (x1 + x2) / 2
					mid_y = (y1 + y2) / 2
					shape = kpt[i, int(mid_x - kernel_len / 2):int(mid_x + kernel_len / 2):,
							int(mid_y - kernel_len / 2):int(mid_y + kernel_len / 2)].shape
					try:
						kpt[i, :shape[0], :shape[1]] = gaussian_kernel[:shape[0], :shape[1]]
					except:
						print('bbox x1,x2,y1,y2:', x1, x2,
							  y1, y2, gaussian_kernel.shape)
						print('yrange:{},{}'.format(
							int((x2 - x1 - y2 + y1) / 2), int((x2 - x1 + y2 - y1) / 2)))
						print('kpt shape:{}'.format(shape))
						traceback.print_exc()
				else:
					kpt[i, x1:x2, y1:y2] = 1.0 * self.loc_weight
				kpt[-1, x1:x2, y1:y2] = 0.0
			# po    s_img (cv_pos_img)
			# ---------

		if self.save_pos and get_fake:
			if self.train:
				img.save(os.path.join(self.save_pos, 'train', self.aug_type,
									  '{}_{}.jpg'.format(index, object_categories[get_fake])), format='JPEG')
			else:
				img.save(os.path.join(self.save_pos, 'test', self.aug_type,
									  '{}_{}.jpg'.format(index, object_categories[get_fake])), format='JPEG')
			if index > 1000:
				sys.exit()

		if self.h_flip:
			if np.random.random() < 0.5:
				img = hflip(img)
				kpt = np.flip(kpt, 2).copy()
			# kpt = kpt[:, ::-1, :].copy()

		if self.v_flip:
			if np.random.random() < 0.5:
				# img = img.transpose(Image.FLIP_LEFT_RIGHT)
				img = vflip(img)
				kpt = np.flip(kpt, 1).copy()
			# kpt = kpt[:, ::-1, :].copy()

		if self.rotate > 0:
			angle = np.random.random() * self.rotate
			angle *= -1 if np.random.random() < 0.5 else 1
			mask = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
			mask_img = Image.fromarray(mask, mode='RGB')
			mask_img = mask_img.rotate(angle)
			mask = 1 - np.asarray(mask_img)
			img = img.rotate(angle)
			img = Image.fromarray(
				(mask * 255 + np.asarray(img)).astype(np.uint8), mode='RGB')
			if len(bboxes):
				for i in range(self.num_classes + 1):
					tmp_img = Image.fromarray(kpt[i].astype(np.uint8).transpose(), mode='L')
					tmp_img = tmp_img.rotate(angle)
					kpt[i] = np.array(tmp_img).transpose()
				mask = np.ones((self.img_size // self.scale, self.img_size // self.scale), dtype=np.uint8)
				mask_img = Image.fromarray(mask, mode='L')
				mask_img = mask_img.rotate(angle)
				mask = 1 - np.asarray(mask_img)
				kpt[-1] += mask
				kpt = kpt.astype(np.float32)

		if self.crop_size > 0:
			start_w = int((self.img_size - self.crop_size) * np.random.random())
			start_h = int((self.img_size - self.crop_size) * np.random.random())
			img = img.crop((start_w, start_h, start_w + self.crop_size,
							start_h + self.crop_size))
			start_w = int(start_h / self.scale)
			start_h = int(start_w / self.scale)
			kpt = kpt[:, start_w:start_w + self.crop_size // self.scale, start_h:start_h + self.crop_size // self.scale]
		# if True:
		#    img.save(f'debug/{index}.jpg')
		#    for i in range(5):
		#        tmp_image = Image.fromarray(kpt[i].astype(np.uint8).transpose()*255, mode='L')
		#        tmp_image = tmp_image.resize((self.img_size, self.img_size))
		#        tmp_image.save(f'debug/{index}_{i}.jpg')

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		for bbox in bboxes:
			bbox[1] /= width
			bbox[2] /= height
			bbox[3] /= width
			bbox[4] /= height
		return (img, os.path.join(self.path_images, f'{path}.jpg')), target, bboxes, torch.Tensor(kpt)

	def __len__(self):
		return len(self.images)

	def get_number_classes(self):
		return self.num_classes
