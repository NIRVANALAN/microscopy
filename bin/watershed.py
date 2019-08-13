import cv2
import os
import sys
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

masks = ['Cytoskeleton', 'Desmosome', 'LipidDroplet', 'MitochondriaDark', 'MitochondriaLight', 'NuclearMembrane']


def watershed(img, save=None):
	# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('img', 600, 600)
	# cv2.namedWindow('fg', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('fg', 600, 600)
	# cv2.namedWindow('bg', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('bg', 600, 600)
	# cv2.namedWindow('uk', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('uk', 600, 600)
	# cv2.namedWindow('jet', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('jet', 600, 600)
	# cv2.namedWindow('res', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('res', 600, 600)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

	# cv2.imshow('img', img)

	# noise removal
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

	# BG
	sure_bg = cv2.dilate(opening, kernel=kernel, iterations=3)
	# cv2.imshow('bg', sure_bg)

	# FG
	distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	# cv2.imshow('img', distance_transform)
	res, sure_fg = cv2.threshold(distance_transform, 0.2 * distance_transform.max(), 255, 0)

	# unknown
	sure_fg = np.uint8(sure_fg)
	# cv2.imshow('fg', sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)
	# cv2.imshow('uk', unknown)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers + 1
	# Now, mark the region of unknown with zero
	markers[unknown == 255] = 0

	# color marter
	# im_color = cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET)
	# cv2.imshow('jet', im_color)

	markers = cv2.watershed(img, markers)
	print(f'organelle number: {np.max(markers) - 1}')
	if save:
		np.save(f'{save}', markers)
		cv2.normalize(markers, markers, 0, 255, cv2.NORM_MINMAX)
		im_color = cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET)
		cv2.imwrite(f'{save}_jit.png', im_color)


# raw_img[markers == 50] = [0, 0, 255]
# cv2.imshow('res', raw_img)

def save_segment_organelle(root_dir, name):
	files = os.listdir(root_dir)
	# if 'png' not in files[0]:  # read TIFF
	# 	for i in files:
	# 		slide_dir = os.path.join(root_dir, i)
	# 		slides = os.listdir(slide_dir)
	# 		for slide in slides:
	# 			slide_files = os.path.join(slide_dir, slide)
	# 			for mask in masks:
	# 				organelle_path = os.path.join(slide_dir, f'{mask}_{i}')
	# 				if f'{slide.split(".")[0]}_{mask}.tiff' in slides:
	# 					tiff_slide = np.array(Image.open(f'{slide}_{mask}.tiff'))
	# 					tiff_slide = cv2.cvtColor(tiff_slide, cv2.COLOR_RGB2BGR)
	# 					watershed(tiff_slide, save=organelle_path)
	#
	# 	pass
	# else:
	for i in masks:
		organelle_path = os.path.join(root_dir, f'{name}_{i}')
		if f'{name}_{i}.png' in files:
			watershed(cv2.imread(f'{organelle_path}.png'), save=organelle_path)
	pass


# watershed(cv2.imread('D:/repos/UTexas/microscopy/microscopy/data/T4/NA_T4_122117_15_Desmosome.png'))

# root = 'D:/repos/UTexas/microscopy/microscopy/data/T4R'
root = 'D:/repos/UTexas/microscopy/microscopy/data/S1'
raw = 'S1_Helios_1of3_v1010.png'
mito_dark: str = 'S1_Helios_1of3_v1010_MitochondriaDark.png'
mito_light: str = 'S1_Helios_1of3_v1270_MitochondriaLight.png'
cytoskeleton: str = 'S1_Helios_1of3_v1010_Cytoskeleton.png'
nuclear_mem: str = 'S1_Helios_1of3_v1010_NuclearMembrane.png'
desmo: str = 'S1_Helios_1of3_v1270_Desmosome.png'
plasma: str = 'S1_Helios_1of3_v1270_PlasmaMembrane.png'
droplet: str = 'NA_T4_122117_11_LipidDroplet.png'
sample = os.path.join(root, droplet)
raw_img = cv2.imread(os.path.join(root, raw))
marker = np.load(os.path.join(root, 'S1_Helios_1of3_v1010_MitochondriaDark.npy'))

# cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('raw', 600, 600)
raw_img[marker >= 2] = [0, 0, 255]
# cv2.imshow('raw', raw_img)
cv2.imwrite('res.jpg', raw_img)

# img = cv2.imread(sample)

# T4R = 'D:/repos/UTexas/microscopy/data/T4R'
# T4R_slides = os.listdir(T4R)

# save_segment_organelle(root, 'NA_T4R_122117_17')

cv2.waitKey()
cv2.destroyAllWindows()

# plt.imshow()
