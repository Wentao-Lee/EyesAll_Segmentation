import torch
import numpy as np

class DiceLoss():
    def __call__(self, pred, target):
        pred = pred.squeeze(dim=1).float()
        # target = class2one_hot(target.long(), 8).float()

        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
        target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)
    return (1 - dice).mean()

class CrossEntropy():
    def __call__(self, pred, target):
        pred = pred.squeeze(dim=1).float()
        torch.nn.CrossEntropyLoss()