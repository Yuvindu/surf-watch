import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def build_segformer_model(
    model_name: str,
    num_classes: int,
    load_pretrained_weights: bool = True,
) -> nn.Module:
    if load_pretrained_weights:
        return SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    config = SegformerConfig(
        num_labels=num_classes,
        id2label={0: "background", 1: "rip"},
        label2id={"background": 0, "rip": 1},
    )
    return SegformerForSemanticSegmentation(config)
