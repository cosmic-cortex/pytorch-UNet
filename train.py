import os

import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
from functools import partial

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import jaccard_index, f1_score, LogNLLLoss
from unet.dataset import Transform2D, ImageToImage2D, Image2D


parser = ArgumentParser()
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--depth', default=5, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--save_model', default=0, type=int)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--learning_rate', type=float, default=1e-3)
args = parser.parse_args()

crop = None
tf_train = Transform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = Transform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)

conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
unet = UNet2D(3, 2, conv_depths)
loss = LogNLLLoss()
optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

results_folder = os.path.join(args.checkpoint_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

weights = [0.0, 1.0]
metric_list = MetricList({'jaccard': partial(jaccard_index, weights=weights),
                          'f1': partial(f1_score, weights=weights)})

model = Model(unet, loss, optimizer, results_folder, device=args.device, save_model=args.save_model)

model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq,
                  predict_dataset=predict_dataset,
                  metric_list=metric_list, verbose=True)
