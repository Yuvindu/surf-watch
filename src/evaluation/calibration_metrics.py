from __future__ import annotations

import numpy as np


NLL_EPSILON = 1e-7


def validate_probability_map(probability_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    probabilities = np.asarray(probability_map, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != shape:
        raise ValueError("probability map must match the 2D target shape")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability map must contain only finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probability map must be within [0, 1]")
    return probabilities


def validate_target(target: np.ndarray) -> np.ndarray:
    labels = np.asarray(target)
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError("target must be a non-empty 2D array")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("target must be binary")
    return labels.astype(np.uint8, copy=False)


def foreground_candidate_mask(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    threshold: float,
) -> np.ndarray:
    labels = validate_target(target)
    first_map = validate_probability_map(first, labels.shape)
    second_map = validate_probability_map(second, labels.shape)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    return (labels == 1) | (first_map >= threshold) | (second_map >= threshold)


class CalibrationAccumulator:
    """Pixel-weighted binary foreground reliability and proper scoring rules."""

    def __init__(self, bin_count: int = 15) -> None:
        if bin_count < 2:
            raise ValueError("bin_count must be at least 2")
        self.bin_count = bin_count
        self.counts = np.zeros(bin_count, dtype=np.int64)
        self.probability_sums = np.zeros(bin_count, dtype=np.float64)
        self.positive_sums = np.zeros(bin_count, dtype=np.float64)
        self.brier_sum = 0.0
        self.nll_sum = 0.0
        self.entropy_sum = 0.0

    def update(
        self,
        probability_map: np.ndarray,
        target: np.ndarray,
        selection: np.ndarray | None = None,
    ) -> None:
        labels = validate_target(target)
        probabilities = validate_probability_map(probability_map, labels.shape)
        if selection is not None:
            selected = np.asarray(selection)
            if selected.shape != labels.shape or selected.dtype != np.bool_:
                raise ValueError("selection must be a boolean mask matching the target")
            probabilities = probabilities[selected]
            labels = labels[selected]
        if probabilities.size == 0:
            return

        probabilities = probabilities.ravel()
        labels = labels.ravel().astype(np.float64, copy=False)
        bins = np.minimum((probabilities * self.bin_count).astype(np.intp), self.bin_count - 1)
        self.counts += np.bincount(bins, minlength=self.bin_count)
        self.probability_sums += np.bincount(
            bins, weights=probabilities, minlength=self.bin_count
        )
        self.positive_sums += np.bincount(bins, weights=labels, minlength=self.bin_count)

        self.brier_sum += float(np.sum((probabilities - labels) ** 2))
        clipped = np.clip(probabilities, NLL_EPSILON, 1.0 - NLL_EPSILON)
        self.nll_sum -= float(
            np.sum(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
        )
        self.entropy_sum -= float(
            np.sum(
                probabilities * np.log(np.clip(probabilities, NLL_EPSILON, 1.0))
                + (1.0 - probabilities)
                * np.log(np.clip(1.0 - probabilities, NLL_EPSILON, 1.0))
            )
        )

    def merge(self, other: CalibrationAccumulator) -> None:
        if self.bin_count != other.bin_count:
            raise ValueError("cannot merge accumulators with different bin counts")
        self.counts += other.counts
        self.probability_sums += other.probability_sums
        self.positive_sums += other.positive_sums
        self.brier_sum += other.brier_sum
        self.nll_sum += other.nll_sum
        self.entropy_sum += other.entropy_sum

    def summary(self) -> dict:
        pixel_count = int(self.counts.sum())
        bins = []
        ece = 0.0
        for index, count in enumerate(self.counts):
            count = int(count)
            mean_probability = float(self.probability_sums[index] / count) if count else None
            foreground_frequency = float(self.positive_sums[index] / count) if count else None
            if count:
                ece += count * abs(mean_probability - foreground_frequency)
            bins.append(
                {
                    "lower": index / self.bin_count,
                    "upper": (index + 1) / self.bin_count,
                    "pixel_count": count,
                    "mean_probability": mean_probability,
                    "foreground_frequency": foreground_frequency,
                }
            )
        return {
            "pixel_count": pixel_count,
            "positive_pixel_count": int(self.positive_sums.sum()),
            "foreground_prevalence": (
                float(self.positive_sums.sum() / pixel_count) if pixel_count else None
            ),
            "brier_score": self.brier_sum / pixel_count if pixel_count else None,
            "negative_log_likelihood": self.nll_sum / pixel_count if pixel_count else None,
            "mean_entropy_nats": self.entropy_sum / pixel_count if pixel_count else None,
            "foreground_ece": ece / pixel_count if pixel_count else None,
            "bins": bins,
        }


class DisagreementAccumulator:
    def __init__(self) -> None:
        self.pixel_count = 0
        self.absolute_difference_sum = 0.0
        self.mask_disagreement_count = 0

    def update(
        self,
        first: np.ndarray,
        second: np.ndarray,
        threshold: float,
        selection: np.ndarray | None = None,
    ) -> None:
        first_array = np.asarray(first)
        if first_array.ndim != 2 or first_array.size == 0:
            raise ValueError("component probability maps must be non-empty 2D arrays")
        first_map = validate_probability_map(first_array, first_array.shape)
        second_map = validate_probability_map(second, first_array.shape)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if selection is not None:
            selected = np.asarray(selection)
            if selected.shape != first_array.shape or selected.dtype != np.bool_:
                raise ValueError("selection must be a boolean mask matching the maps")
            first_map = first_map[selected]
            second_map = second_map[selected]
        self.pixel_count += int(first_map.size)
        self.absolute_difference_sum += float(np.sum(np.abs(first_map - second_map)))
        self.mask_disagreement_count += int(
            np.count_nonzero((first_map >= threshold) != (second_map >= threshold))
        )

    def merge(self, other: DisagreementAccumulator) -> None:
        self.pixel_count += other.pixel_count
        self.absolute_difference_sum += other.absolute_difference_sum
        self.mask_disagreement_count += other.mask_disagreement_count

    def summary(self) -> dict:
        return {
            "pixel_count": self.pixel_count,
            "mean_absolute_probability_difference": (
                self.absolute_difference_sum / self.pixel_count if self.pixel_count else None
            ),
            "mask_disagreement_fraction": (
                self.mask_disagreement_count / self.pixel_count if self.pixel_count else None
            ),
        }
