import os
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from dataset import Transform2D, ImageToImage2D


parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()

transform = ImageToImage2D(args.dataset)
