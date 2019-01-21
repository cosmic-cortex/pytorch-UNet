import torch
from operator import mul
from functools import reduce


def jaccard_index(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, n_height, n_width)
        gt: torch.LongTensor of shape (n_batch, n_height, n_width)
    """

    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = torch.max(output, gt)

    return (intersection.sum().float()/union.sum()).item()


def accuracy(output, gt):
    output = torch.argmax(output, dim=1, keepdim=True)      # determining the predictions
    correct = (output == gt).sum().float().item()
    all = reduce(mul, output.shape)

    return correct/all
