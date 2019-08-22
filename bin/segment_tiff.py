import numpy as np
import random
import cv2
import os
import os.path as osp
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
mask_root = 'D://repo//UTexas//microscopy//extracted_data//'

extracted_root = 'D:/repo/UTexas/microscopy/extracted_data'
raw_dir = osp.join(extracted_root, 'raw')
label_dir = osp.join(extracted_root, 'labels')

# a = tiff.imread(s1)
# a = np.uint16(a)
# print(a.shape)
# print(a.dtype)
pass

# patch_size = 512
num_classes = 8


def mkdir_if_not(file_dir):
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def slideing_window_crop(save_dir, patch_size=64, slide_patch_ratio=0.125, threshold=0.4,
                         extra_threshold=0.1):  # for membrane and desmosome
    save_root_dir = osp.join(save_dir, str(patch_size), str(threshold))
    if not osp.isdir(save_root_dir):
        os.makedirs(save_root_dir)
    raw_slide = os.listdir(raw_dir)
    for slide in raw_slide:
        print(f'deal with {slide}')
        slide_name, suffix = slide.split('.')
        save_slide_dir = osp.join(save_root_dir, slide_name)
        save_slide_raw_dir = osp.join(save_slide_dir, 'image')
        save_slide_label_dir = osp.join(save_slide_dir, 'label')
        mkdir_if_not(save_slide_dir)
        mkdir_if_not(save_slide_raw_dir)
        mkdir_if_not(save_slide_label_dir)
        raw_tiff = tiff.imread(osp.join(raw_dir, slide))
        label = np.load(osp.join(label_dir, slide.replace(suffix, 'npy')))
        x, y, stride = 0, 0, int(patch_size * slide_patch_ratio)
        shape = raw_tiff.shape
        img_num = 0
        while (x + patch_size) < shape[0] and (y + patch_size) < shape[1]:
            tile_label = label[x:x + patch_size, y:y + patch_size]
            proportion = np.count_nonzero(tile_label) / tile_label.size
            detail_proportion = np.array([np.count_nonzero(tile_label == i) for i in (1, 2, 5)]).sum() / tile_label.size
            if proportion >= threshold or detail_proportion >= extra_threshold:
                print(f'detail_proportion:{detail_proportion}') if detail_proportion >= extra_threshold else None
                print(f'x:{x} y:{y}  proportion:{proportion}')
                tile_img = raw_tiff[x:x + patch_size, y:y + patch_size]
                np.save(osp.join(save_slide_raw_dir, f'{x}_{y}'), tile_img)
                np.save(osp.join(save_slide_label_dir, f'{x}_{y}'), tile_label)
                img_num += 1
            x += stride
            y += stride
        print(f'{img_num} images saved')


def generate_all_mask(save_dir, dataset='D://repos//UTexas//microscopy//data'):
    datasets = [os.path.join(dataset, i) for i in ('s1', 't4', 't4r')]
    for _ in datasets:
        if not os.path.isdir(_):
            os.makedirs(_)
        slides = os.listdir(_)
        dataset = _
        for slide in slides:
            if 's1' in slide:
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


def generate_list(dataset_root):
    dataset_patch = os.listdir(dataset_root)
    for patch in dataset_patch:
        patch_root = osp.join(dataset_root, patch)
        threshold = os.listdir(patch_root)
        for ts in threshold:
            threshold_dir = osp.join(patch_root, ts)
            slides = os.listdir(threshold_dir)
            all_list = []
            for slide in slides:
                if 'txt' in slide:
                    continue
                cell_type = slide.split('_')[1] if 'T4' in slide else 'S1'
                relative_root = osp.join(patch, ts)
                slide_img_root = osp.join(relative_root, slide, 'image')
                slide_label_root = osp.join(relative_root, slide, 'label')
                imgs = os.listdir(osp.join(dataset_root, slide_img_root))
                slide_img = [
                    (os.path.join(slide_img_root, img).replace("\\", "/")) + ';' + (osp.join(slide_label_root, img).replace("\\", "/")) + f';{cell_type}\n'
                    for img in imgs]
                with open(os.path.join(threshold_dir, slide, 'all_list.txt'), 'w') as filehandle:
                    filehandle.writelines(slide_img)
                all_list.extend(slide_img)
            print(ts)
            random.shuffle(all_list)
            train_list = all_list[:int(0.8 * all_list.__len__())]
            test_list = all_list[train_list.__len__():]
            with open(os.path.join(threshold_dir, f'all_list.txt'), 'w') as filehandle:
                filehandle.writelines(all_list)
                print(f'total_list len: {all_list.__len__()}')
            with open(os.path.join(threshold_dir, f'train_list.txt'), 'w') as filehandle:
                filehandle.writelines(train_list)
                print(f'train_list len: {train_list.__len__()}')
            with open(os.path.join(threshold_dir, f'test_list.txt'), 'w') as filehandle:
                filehandle.writelines(test_list)
                print(f'test_list len: {test_list.__len__()}')

    pass


# patch_size = 1024
save_root = os.path.join('D://repo//UTexas//dataset')

# for ps in [64, 128]:
#     for ts in [0.3, 0.5]:
#         slideing_window_crop(save_dir=save_root, patch_size=ps, threshold=ts)

generate_list(save_root)

# generate_all_mask(save_root)
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
