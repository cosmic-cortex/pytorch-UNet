import os

from argparse import ArgumentParser

from unet.model import Model
from unet.dataset import Image2D

parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()

predict_dataset = Image2D(args.dataset)
model = torch.load(args.model_path)

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)

model = Model(unet, checkpoint_folder=args.results_path, device=args.device)

model.predict_dataset(predict_dataset, args.result_path)