from typing import Tuple

import cv2
import numpy as np
import torch

from configs.baseline_config import BaselineConfig
from src.models.segformer_baseline import build_segformer_model
from src.models.segmentation_interface import (
    SegmentationModelMetadata,
    SegmentationResult,
)


class SegFormerSegmentationAdapter:
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
        self.threshold = threshold
        self.image_size = self._validate_image_size(config.image_size)
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        model = build_segformer_model(
            model_name=self.config.pretrained_model_name,
            num_classes=self.config.num_classes,
            load_pretrained_weights=False,
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _validate_image_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        if len(image_size) != 2:
            raise ValueError("image_size must be a (height, width) tuple")

        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size height and width must be positive")

        return height, width

    def metadata(self) -> SegmentationModelMetadata:
        return SegmentationModelMetadata(
            name="SegFormer",
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
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        self._validate_frame(frame)
        input_tensor = self.preprocess_frame(frame).to(self.device)

        logits = self.model(pixel_values=input_tensor).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(logits, dim=1)
        rip_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy().astype(np.float32)
        return cv2.resize(
            rip_prob,
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

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("frame cannot be None")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (height, width, 3)")

        if frame.dtype != np.uint8:
            raise ValueError("frame must use dtype uint8")
