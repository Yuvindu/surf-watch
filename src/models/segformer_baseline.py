import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def build_segformer_model(model_name: str, num_classes: int) -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model