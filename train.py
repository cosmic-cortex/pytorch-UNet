import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from unet.dataset import Transform2D, ImageToImage2D


parser = ArgumentParser()
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
args = parser.parse_args()

tf_train = Transform2D(crop=(512, 512), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                       p_random_affine=0.5, long_mask=True)
tf_val = Transform2D(crop=(512, 512), p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)

unet = UNet2D(1, 2, [10, 20, 30, 40])
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(unet.parameters(), lr=1e-3)

model_name = '2019-01-18-test'
file_dir = os.path.split(os.path.realpath(__file__))[0]
results_folder = os.path.join(file_dir, '..', 'results', model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

model = Model(unet, loss, optimizer, results_folder, device=args.device)
model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq, verbose=True)

