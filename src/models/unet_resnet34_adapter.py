from typing import Tuple

import cv2
import numpy as np
import torch

from configs.baseline_config import BaselineConfig
from src.models.segmentation_interface import (
    SegmentationModelMetadata,
    SegmentationResult,
)
from src.models.unet_resnet34 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_unet_resnet34_model,
)


class UnetResNet34SegmentationAdapter:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        config: BaselineConfig,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.config = config
        self.threshold = self._validate_threshold(threshold)
        self.image_size = self._validate_image_size(config.image_size)
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        # The trained checkpoint contains encoder weights, so inference must not download them.
        model = build_unet_resnet34_model(encoder_weights=None).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                "U-Net checkpoint must be a dictionary containing 'model_state_dict'."
            )

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the U-Net ResNet34 adapter."
            ) from exc
        model.eval()
        return model

    def metadata(self) -> SegmentationModelMetadata:
        return SegmentationModelMetadata(
            name="U-Net (ResNet34)",
            input_size=self.image_size,
            threshold=self.threshold,
            device=str(self.device),
        )

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        self._validate_frame(frame)
        resized = cv2.resize(
            frame,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    @torch.no_grad()
    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        self._validate_frame(frame)
        input_tensor = self.preprocess_frame(frame).to(self.device)
        logits = self.model(input_tensor)

        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError(
                "U-Net must return logits with shape (1, 1, height, width)."
            )

        if tuple(logits.shape[-2:]) != self.image_size:
            logits = torch.nn.functional.interpolate(
                logits,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

        probability_map = torch.sigmoid(logits).squeeze(0).squeeze(0)
        probability_map = probability_map.cpu().numpy().astype(np.float32)
        return cv2.resize(
            probability_map,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

    def probability_to_mask(self, probability_map: np.ndarray) -> np.ndarray:
        if probability_map.ndim != 2:
            raise ValueError("probability_map must be a 2D array")
        return (probability_map >= self.threshold).astype(np.uint8)

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        probability_map = self.predict_probability_map(frame)
        binary_mask = self.probability_to_mask(probability_map)
        return SegmentationResult(
            probability_map=probability_map,
            binary_mask=binary_mask,
            metadata=self.metadata(),
        )

    def _validate_image_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        if len(image_size) != 2:
            raise ValueError("image_size must be a (height, width) tuple")

        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size height and width must be positive")
        return height, width

    def _validate_threshold(self, threshold: float) -> float:
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return value

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("frame cannot be None")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (height, width, 3)")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use dtype uint8")
