import torch


def jaccard_index(output, gt, long_gt=True):
    if long_gt:
        gt = torch.zeros(output.shape).scatter_(1, gt, torch.ones(output.shape))

    intersection = output*gt
    union = torch.max(output, gt)

    return intersection.sum()/union.sum()


def accuracy(output, gt):
    pass


def f1_score(output, gt):
    pass
