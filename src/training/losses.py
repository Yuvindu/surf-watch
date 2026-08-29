import torch.nn as nn


def get_loss_fn():
    return nn.CrossEntropyLoss()


def get_binary_loss_fn():
    return nn.BCEWithLogitsLoss()
