import os

import torch
from PIL import Image, ImageDraw
import numpy as np

from src.training.metrics import (
    compute_confusion,
    iou_score,
    dice_score,
    precision_score,
    recall_score,
)
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
    mean_precision, _ = precision_score(confusion)
    mean_recall, _ = recall_score(confusion)

    return {
        "val_loss": running_loss / max(len(loader), 1),
        "val_iou": mean_iou,
        "val_dice": mean_dice,
        "val_precision": mean_precision,
        "val_recall": mean_recall,
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
            filenames = [f"sample_{i}.jpg" for i in range(images.shape[0])]

        images = images.to(device)
        outputs = model(pixel_values=images).logits
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.argmax(outputs, dim=1).cpu()
        images_cpu = images.cpu()
        masks_cpu = masks.cpu()

        for i in range(images.shape[0]):
            if saved >= max_samples:
                return

            image_np = (images_cpu[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (masks_cpu[i].numpy() * 255).astype(np.uint8)
            pred_np = (preds[i].numpy() * 255).astype(np.uint8)

            gt_rgb = np.stack([gt_np] * 3, axis=-1)
            pred_rgb = np.stack([pred_np] * 3, axis=-1)

            comparison = np.concatenate([image_np, gt_rgb, pred_rgb], axis=1)
            comparison_img = Image.fromarray(comparison)

            label_height = 30
            labeled_img = Image.new(
                "RGB",
                (comparison_img.width, comparison_img.height + label_height),
                color=(0, 0, 0),
            )
            labeled_img.paste(comparison_img, (0, label_height))

            draw = ImageDraw.Draw(labeled_img)
            panel_width = image_np.shape[1]
            labels = ["Input", "Ground Truth", "Prediction"]

            for panel_idx, label in enumerate(labels):
                x = panel_idx * panel_width + 10
                y = 8
                draw.text((x, y), label, fill=(255, 255, 255))

            base_name = os.path.splitext(filenames[i])[0]
            out_path = os.path.join(output_dir, f"compare_{base_name}.png")
            labeled_img.save(out_path)
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