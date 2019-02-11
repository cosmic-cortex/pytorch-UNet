import os
import numpy as np
import torch

from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    def __init__(self, crop=(256, 256), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage2D(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |-- images
          |-- img001.png
          |-- img002.png
          |-- ...
      |-- masks
          |-- img001.png
          |-- img002.png
          |-- ...

    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)

        self.joint_transform = joint_transform
        self.one_hot_mask = one_hot_mask

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # read image
        image = io.imread(os.path.join(self.input_path, image_filename))
        # read mask image
        mask = io.imread(os.path.join(self.output_path, image_filename))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class Image2D(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |-- input
          |-- img001.png
          |-- img002.png
          |-- ...
    """

    def __init__(self, dataset_path: str, transform: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.input_path)

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))

        # correct dimensions if needed
        image = correct_dims(image)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        return image, image_filename
