import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

category = ['PlasmaMembrane', 'NuclearMembrane', 'MitochondriaDark', 'MitochondriaLight', 'Desmosome', 'Cytoskeleton',
			'LipidDroplet']
raws = ['S1', 'T4', 'T4R']
root = 'D://repos//UTexas//microscopy//data//'
patch_size = 512
save_dirs = [os.path.join('D://repos//UTexas//filtered_dataset', str(patch_size), i) for i in raws]


def generate_raw_data():
	datasets = [os.path.join(root, i) for i in raws]
	for i in range(len(raws)):
		save_dir = save_dirs[i]
		dataset = datasets[i]
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
			slide_no = tiff_files
			for i in tiff_files:
				if '.tif' in i:
					print(f'deal with {i}')
					raw_tiff = plt.imread(os.path.join(slide_dir, i))
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


def generate_mask(raw_data_dir, shape=484):
	datasets = [os.path.join(raw_data_dir, i) for i in raws]
	for dataset in datasets:  # 'S1,T4,T4R'
		dirs = os.listdir(dataset)
		for data in dirs:
			slides = os.listdir(
				os.path.join(dataset, data))  # go in a slide of T4/T4R/S1, ['NA_T4_122117_01','NA_T4_122117_11']
			if 'S1' in dataset:
				slide_dirs = [os.path.join(dataset, data, f'{data}{i}') for i in ['010', '270']]
				# all_slides = os.listdir(os.path.join(dataset, data))
				for slide_dir in slide_dirs:
					bg_img_num = 0
					if not os.path.isdir(f'{slide_dir}_label'):
						os.makedirs(f'{slide_dir}_label')
					raw_slide_name = slide_dir.split('\\')[-1]
					shapes = (shape, shape)
					# print(f'masks shape: {masks.shape}')
					for img_name in os.listdir(slide_dir):
						masks = np.zeros((len(category) + 1, *shapes)).astype(np.int_)  # long
						for organelle in range(len(category)):
							if f'{raw_slide_name}_{category[organelle]}' in slides:
								mask = load_pil(os.path.join(f'{slide_dir}_{category[organelle]}', img_name),
												shape=shape)
								masks[organelle + 1] = mask
						target_mask = np.argmax(masks, 0)  # bg->0
						if np.sum(target_mask) > 0:
							np.save(os.path.join(f'{slide_dir}_label', f'{img_name}'), target_mask)
						else:
							bg_img_num += 1
					print(f'{slide_dir} done, bg number: {bg_img_num}')
			else:
				# slide_dirs = [os.path.join(dataset, data, f'{data}{i}') for i in ['010', '270']]
				# all_slides = os.listdir(os.path.join(dataset, data))
				# for slide_dir in slide_dirs:
				slide_dir = os.path.join(dataset, data, data)
				bg_img_num = 0
				if not os.path.isdir(f'{slide_dir}_label'):
					os.makedirs(f'{slide_dir}_label')
				raw_slide_name = slide_dir.split('\\')[-1]
				shapes = (shape, shape)
				# print(f'masks shape: {masks.shape}')
				for img_name in os.listdir(slide_dir):
					masks = np.zeros((len(category) + 1, *shapes)).astype(np.int_)  # long
					for organelle in range(len(category)):
						if f'{raw_slide_name}_{category[organelle]}' in slides:
							mask = load_pil(os.path.join(f'{slide_dir}_{category[organelle]}', img_name),
											shape=shape)
							masks[organelle + 1] = mask
					target_mask = np.argmax(masks, 0)  # bg->0
					if np.sum(target_mask) > 0:
						np.save(os.path.join(f'{slide_dir}_label', f'{img_name}'), target_mask)
					else:
						bg_img_num += 1
				print(f'{slide_dir} done, bg number: {bg_img_num}')


# dataset = raw_data_dir[i]
# if not os.path.isdir(save_dir):
# 	os.makedirs(save_dir)
# slides = os.listdir(dataset)


generate_mask(os.path.join('D://repos//UTexas//dataset//', '512'))

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
