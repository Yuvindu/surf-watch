import torch


def compute_confusion(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2):
    preds = preds.view(-1)
    targets = targets.view(-1)

    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)
    indices = num_classes * targets + preds
    conf += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


def iou_score(confusion: torch.Tensor):
    tp = torch.diag(confusion).float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou.mean().item(), iou


def dice_score(confusion: torch.Tensor):
    tp = torch.diag(confusion).float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    denom = 2 * tp + fp + fn
    dice = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(tp))
    return dice.mean().item(), dice