import os
import numpy as np
import torch

from skimage import io

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


class Transform2D:
    def __init__(self, crop=(256, 256), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, normalize=False, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.normalize = normalize
        self.long_mask = long_mask

    def __call__(self, input, output):
        # transforming to PIL image
        input, output = F.to_pil_image(input), F.to_pil_image(output)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(input, self.crop)
            input, output = F.crop(input, i, j, h, w), F.crop(output, i, j, h, w)
            if np.random.rand() < self.p_flip:
                input, output = F.hflip(input), F.hflip(output)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            input = self.color_tf(input)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)


        # transforming to tensor
        input = F.to_tensor(input)
        if not self.long_mask:
            output = F.to_tensor(output)
        else:
            output = to_long_tensor(output)

        # normalizing image
        if self.normalize:
            input = tf_normalize(input)

        return input, output


class ImageToImage2D(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |-- input
          |-- img001.png
          |-- img002.png
          |-- ...
      |-- output
          |-- img001.png
          |-- img002.png
          |-- ...

    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'input')
        self.output_path = os.path.join(dataset_path, 'output')
        self.images_list = os.listdir(self.input_path)

        self.joint_transform = joint_transform
        self.one_hot_mask = one_hot_mask

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        img_in = io.imread(os.path.join(self.input_path, image_filename))

        # read output image in training mode
        img_out = io.imread(os.path.join(self.output_path, image_filename))
        # TODO: rewrite this such that adding dimensions is optional
        if len(img_out.shape) == 2:
            img_out = np.expand_dims(img_out, axis=2)

        if self.joint_transform:
            img_in, img_out = self.joint_transform(img_in, img_out)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            img_out = torch.zeros((self.one_hot_mask, img_out.shape[1], img_out.shape[2])).scatter_(0, img_out.long(), 1)

        return img_in, img_out, image_filename


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
        self.input_path = os.path.join(dataset_path, 'input')
        self.images_list = os.listdir(self.input_path)

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.input_path, image_filename))

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image, image_filename