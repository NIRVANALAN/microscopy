import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image

category = ['bg', 'PlasmaMembrane', 'NuclearMembrane', 'MitochondriaDark', 'MitochondriaLight', 'Desmosome',
			'Cytoskeleton',
			'LipidDroplet']
raws = ['S1', 'T4', 'T4R']
root = 'D://repos//UTexas//microscopy//data//'
s1 = '../data/S1_Helios_1of3_v1270.tiff'
t4r = '../data/CL_T4R_180807_06.tif'

# a = tiff.imread(s1)
# a = np.uint16(a)
# print(a.shape)
# print(a.dtype)
pass

# patch_size = 512
num_classes = 8


def generate_all_mask(save_dir, dataset='D://repos//UTexas//microscopy//data'):
	datasets = [os.path.join(dataset, i) for i in ('S1', 'T4', 'T4R')]
	for _ in datasets:
		if not os.path.isdir(_):
			os.makedirs(_)
		slides = os.listdir(_)
		dataset = _
		for slide in slides:
			if 'S1' in slide:
				raw = [os.path.join(dataset, slide, f'{slide}{i}') for i in ['010.tiff', '270.tiff']]
			# raw = tiff.imread(os.path.join(dataset, slide, slide + '.tiff'))
			else:
				raw = [(os.path.join(dataset, slide, slide + '.tif'))]
			for raw_data_name in raw:
				raw_data_array = tiff.imread(raw_data_name)
				gt_mask = np.ndarray(shape=(num_classes, *raw_data_array.shape), dtype='uint16')
				print(raw_data_name)
				for i in category:
					try:
						organ = tiff.imread(os.path.join(dataset, slide, f'{raw_data_name.split(".")[0]}_{i}.tiff'))
						if organ.dtype != 'uint16':
							organ = np.uint16(organ)
						gt_mask[category.index(i)] = organ
						print(i)
					except:
						continue
				gt_mask = gt_mask.argmax(0)
				print(gt_mask.max())
				np.save(os.path.join(save_dir, f'{os.path.split(raw_data_name)[-1].split(".")[0]}'),
						gt_mask)


def generate_raw_data(tile_save_root, patch_size=1024):
	raw_data_set = [os.path.join(root, i) for i in raws]
	tile_save_root = [os.path.join(tile_save_root, i) for i in raws]
	for i in range(len(raws)):
		save_dir = tile_save_root[i]
		dataset = raw_data_set[i]
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		slides = os.listdir(dataset)
		for slide in slides:
			print(slide)
			slide_dir = os.path.join(dataset, slide)
			save_slide_dir = os.path.join(save_dir, slide)
			if not os.path.isdir(save_slide_dir):
				os.mkdir(save_slide_dir)
			tiff_files = os.listdir(slide_dir)
			# slide_no = tiff_files
			for i in tiff_files:
				if '.tif' in i:
					print(f'deal with {i}')
					raw_tiff = plt.imread(os.path.join(slide_dir, i)).copy()
					if not '0' <= i.split('.')[0] <= '9':  # organelle
						raw_tiff[raw_tiff == 1] = 255
					raw_tiff_name = i.split('.')[0]
					raw_tiff_dir = os.path.join(save_slide_dir, raw_tiff_name)
					if not os.path.isdir(raw_tiff_dir):
						os.mkdir(raw_tiff_dir)
					X, Y = raw_tiff.shape
					X = X // patch_size
					Y = Y // patch_size
					for x in range(X):
						for y in range(Y):
							tile = Image.fromarray(np.uint8(
								raw_tiff[x * patch_size:(x + 1) * patch_size,
								y * patch_size:(y + 1) * patch_size].copy()))
							tile.save(os.path.join(raw_tiff_dir, f'{x}_{y}.png'))


def load_pil(img, shape=None):
	img = Image.open(img)
	if shape:
		img = img.resize((shape, shape), Image.BILINEAR)
	return np.array(img)


def generate_mask(tile_data_dir, mask_size=1024):
	datasets = [os.path.join(tile_data_dir, i) for i in raws]
	shapes = (mask_size, mask_size)
	for raw_data_set in datasets:  # 'S1,T4,T4R'
		slide_root = os.listdir(raw_data_set)  # root of each slide
		for data in slide_root:
			slides = os.listdir(
				os.path.join(raw_data_set, data))  # go in a slide of T4/T4R/S1, ['NA_T4_122117_01','NA_T4_122117_11']
			if 'S1' in raw_data_set:
				slide_dirs = [os.path.join(raw_data_set, data, f'{data}{i}') for i in ['010', '270']]
			else:
				slide_dirs = [os.path.join(raw_data_set, data, data)]
			# all_slides = os.listdir(os.path.join(raw_data_set, data))
			for slide_dir in slide_dirs:
				bg_img = []
				if not os.path.isdir(f'{slide_dir}_label'):
					os.makedirs(f'{slide_dir}_label')
				raw_slide_name = slide_dir.split('\\')[-1]
				# print(f'masks shape: {masks.shape}')

				# save mask
				for img_name in os.listdir(slide_dir):
					masks = np.zeros((len(category) + 1, *shapes)).astype(np.int_)  # long
					for organelle in range(len(category)):
						if f'{raw_slide_name}_{category[organelle]}' in slides:
							mask = load_pil(os.path.join(f'{slide_dir}_{category[organelle]}', img_name))
							masks[organelle + 1] = mask
					# _ = np.sum(masks, 0)
					# if np.max(_) > 255:
					# 	raise ValueError
					target_mask = np.argmax(masks, 0)  # bg->0
					if np.sum(target_mask) > 0:
						np.save(os.path.join(f'{slide_dir}_label', f'{img_name}'), target_mask)
					else:
						bg_img.append(f'{slide_dir}_{img_name}')
				print(f'{slide_dir} done, bg number: {bg_img.__len__()}, {bg_img}')


# patch_size = 1024
save_root = os.path.join('D://repos//UTexas//dataset//', 'whole_data')
generate_all_mask(save_root)
# # generate_raw_data(tile_save_root=save_root, patch_size=1024)
# generate_mask(tile_data_dir=save_root, mask_size=1024)

# for i in category:
# 	try:
# 		filename = os.path.join(slide_dir, f'{slide}_{i}.tiff')
# 		if os.path.isfile(filename):
# 			organ = plt.imread(os.path.join(dataset, slide, f'{slide}_{i}.tiff'))
# 			organ[organ == 1] = 255
# 			tmp_array[organ == 255] = webcolors.name_to_rgb(color[category.index(i)])
# 			organ_img = Image.fromarray(np.uint8(organ))
# 			print(i)
# 			print(organ.dtype)
# 			organ_img.save(save_dir + slide + f'_{i}.png')
# 		else:
# 			continue
# 	except:
# 		traceback.print_exc()
# 		continue
#
# tmp = Image.fromarray(tmp_array)
# tmp.save(save_dir + slide + f'_organells.png')
