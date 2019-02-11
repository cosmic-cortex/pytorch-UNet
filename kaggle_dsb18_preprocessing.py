import os
import numpy as np

from skimage import io
from shutil import copy
from argparse import ArgumentParser

from unet.utils import chk_mkdir


def merge_masks(masks_folder):
    masks = list()
    for mask_img_filename in os.listdir(masks_folder):
        mask_img = io.imread(os.path.join(masks_folder, mask_img_filename))
        masks.append(mask_img)

    merged_mask = np.sum(masks, axis=0)
    merged_mask[merged_mask > 0] = 1

    return merged_mask


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--export_path', required=True, type=str)
    args = parser.parse_args()

    new_images_folder = os.path.join(args.export_path, 'images')
    new_masks_folder = os.path.join(args.export_path, 'masks')

    chk_mkdir(args.export_path, new_images_folder, new_masks_folder)

    for image_name in os.listdir(args.dataset_path):
        images_folder = os.path.join(args.dataset_path, image_name, 'images')
        masks_folder = os.path.join(args.dataset_path, image_name, 'masks')
        # copy the image
        copy(src=os.path.join(images_folder, image_name + '.png'),
             dst=os.path.join(new_images_folder, image_name + '.png'))

        # convert and save the masks
        mask_img = merge_masks(masks_folder)
        io.imsave(os.path.join(new_masks_folder, image_name + '.png'), mask_img)
