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
datasets = [os.path.join(root, i) for i in raws]
save_dirs = [os.path.join('D://repos//UTexas//dataset', str(patch_size), i) for i in raws]

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
							raw_tiff[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].copy()))
						tile.save(os.path.join(raw_tiff_dir, f'{x}_{y}.png'))

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
