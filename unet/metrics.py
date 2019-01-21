import torch
from operator import mul
from functools import reduce


def jaccard_index(output, gt, long_gt=True):
    if long_gt:
        gt = torch.zeros_like(output).scatter_(1, gt, 1)

    intersection = output*gt
    union = torch.max(output, gt)

    return (intersection.sum().float()/union.sum()).item()


def accuracy(output, gt, long_gt=True):
    output = torch.argmax(output, dim=1, keepdim=True)      # determining the predictions
    if not long_gt:
        gt = gt = torch.argmax(gt, dim=1, keepdim=True)

    return ((output == gt).sum().float()/reduce(mul, output.shape)).item()
