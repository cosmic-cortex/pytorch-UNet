import os
import numpy as np
import torch

from skimage import io

from torch.utils.data import Dataset

from albumentations.pytorch import ToTensor

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


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        aug: augmentation transform from albumentations. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, aug: Callable = None, one_hot_mask: int = False,
                 long_mask: bool = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.long_mask = long_mask

        if aug:
            self.aug = aug
        else:
            self.aug = ToTensor()

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

        if self.aug:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        if self.long_mask:
            # convert mask dtype to long and getting rid of extra dimension
            mask = torch.squeeze(mask.long())

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, aug: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.input_path)

        if aug:
            self.aug = aug
        else:
            self.aug = ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))

        # correct dimensions if needed
        image = correct_dims(image)

        image = self.aug(image=image)['image']

        return image, image_filename
