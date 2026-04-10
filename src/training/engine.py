import os

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF

from src.training.metrics import compute_confusion, iou_score, dice_score
from src.training.utils import save_checkpoint


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, masks, *_rest in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros((2, 2), dtype=torch.int64, device=device)

    for images, masks, *_rest in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = loss_fn(outputs, masks)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        confusion += compute_confusion(preds, masks, num_classes=2)

    mean_iou, _ = iou_score(confusion)
    mean_dice, _ = dice_score(confusion)

    return {
        "val_loss": running_loss / max(len(loader), 1),
        "val_iou": mean_iou,
        "val_dice": mean_dice,
    }


@torch.no_grad()
def save_sample_predictions(model, loader, device, output_dir, max_samples=4):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    saved = 0
    for batch in loader:
        if len(batch) == 3:
            images, masks, filenames = batch
        else:
            images, masks = batch
            filenames = [f"sample_{i}.png" for i in range(images.shape[0])]

        images = images.to(device)
        outputs = model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.argmax(outputs, dim=1).cpu()

        for i in range(images.shape[0]):
            if saved >= max_samples:
                return

            pred = (preds[i].numpy() * 255).astype(np.uint8)
            out_path = os.path.join(output_dir, f"pred_{filenames[i].replace('.jpg', '.png')}")
            Image.fromarray(pred).save(out_path)
            saved += 1


def save_best_model(model, optimizer, epoch, metrics, checkpoint_dir, filename="best_model.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    save_checkpoint(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )