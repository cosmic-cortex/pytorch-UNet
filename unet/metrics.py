import torch
from operator import mul
from functools import reduce


def jaccard_index(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection # torch.max(output, gt)

    return (intersection.sum().float()/union.sum()).item()


def accuracy(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    output = torch.argmax(output, dim=1)      # determining the predictions
    correct = (output == gt).sum().float().item()
    all = reduce(mul, output.shape)

    return correct/all


def f1_score(output, gt, n_classes):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
        n_classes: number of classes
    """

    epsilon = 1e-20

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i)*(gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon)/(selected + epsilon)
    recall = (true_positives + epsilon)/(relevant + epsilon)

    return (2*(precision*recall)/(precision + recall)).mean().item()
