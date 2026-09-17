from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CONFUSION_KEYS = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)


def binary_confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """Return binary pixel counts after validating aligned 2D masks."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    if prediction.ndim != 2:
        raise ValueError("prediction and target must be 2D arrays")

    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    return {
        "true_negative": int(np.count_nonzero(~pred & ~truth)),
        "false_positive": int(np.count_nonzero(pred & ~truth)),
        "false_negative": int(np.count_nonzero(~pred & truth)),
        "true_positive": int(np.count_nonzero(pred & truth)),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def metrics_from_confusion(confusion: dict[str, int]) -> dict[str, float | None]:
    """Compute foreground and class-mean metrics from binary pixel counts.

    Undefined metrics are represented by ``None`` instead of being silently
    scored as zero or one. This keeps all-background frames from inflating a
    model's macro score.
    """
    tn = confusion["true_negative"]
    fp = confusion["false_positive"]
    fn = confusion["false_negative"]
    tp = confusion["true_positive"]

    foreground_iou = _ratio(tp, tp + fp + fn)
    background_iou = _ratio(tn, tn + fp + fn)
    foreground_dice = _ratio(2 * tp, 2 * tp + fp + fn)
    background_dice = _ratio(2 * tn, 2 * tn + fp + fn)

    defined_ious = [
        value for value in (background_iou, foreground_iou) if value is not None
    ]
    defined_dice = [
        value for value in (background_dice, foreground_dice) if value is not None
    ]
    return {
        "foreground_iou": foreground_iou,
        "foreground_dice": foreground_dice,
        "foreground_precision": _ratio(tp, tp + fp),
        "foreground_recall": _ratio(tp, tp + fn),
        "background_iou": background_iou,
        "mean_iou": float(np.mean(defined_ious)) if defined_ious else None,
        "mean_dice": float(np.mean(defined_dice)) if defined_dice else None,
    }


def add_confusions(confusions: Iterable[dict[str, int]]) -> dict[str, int]:
    total = {key: 0 for key in CONFUSION_KEYS}
    for confusion in confusions:
        for key in CONFUSION_KEYS:
            total[key] += int(confusion[key])
    return total


def mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def summarize_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("at least one evaluation record is required")

    confusion = add_confusions(record["confusion"] for record in records)
    metric_names = tuple(metrics_from_confusion(confusion))
    macro = {
        name: mean_defined(record["metrics"][name] for record in records)
        for name in metric_names
    }
    return {
        "frame_count": len(records),
        "confusion": confusion,
        "micro": metrics_from_confusion(confusion),
        "macro_per_frame": macro,
        "empty_ground_truth_frames": sum(
            record["ground_truth_positive_pixels"] == 0 for record in records
        ),
        "empty_prediction_frames": sum(
            record["prediction_positive_pixels"] == 0 for record in records
        ),
        "mean_ground_truth_foreground_fraction": float(
            np.mean([record["ground_truth_foreground_fraction"] for record in records])
        ),
        "mean_prediction_foreground_fraction": float(
            np.mean([record["prediction_foreground_fraction"] for record in records])
        ),
    }
