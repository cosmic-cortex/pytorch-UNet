import os
import numpy as np

from skimage import io

from shutil import copy
from collections import Container
from argparse import ArgumentParser


# I would like to take a minute to express that relative imports in Python are horrible.
# Although this function is implemented somewhere else, it cannot be imported, since its
# folder is in the parent folder of this. Relative imports result in ValueErrors. The
# design choice behind this decision eludes me. The only way to circumvent this is either
# make this package installable, add the parent folder to PATH or implement it again.
# I went with the latter one.
#
# If you are reading this and you also hate the relative imports in Python, cheers!
# You are not alone.
def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


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
