"""MobileNetV2 comparison model (report Fig. 4 names MobileNetV2)."""

from __future__ import annotations

from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from skincancer.config import ModelConfig
from skincancer.models.base import TransferClassifier
from skincancer.preprocessing.transforms import IMAGENET_NORMALIZATION

MOBILENET_V2_CHANNELS = 1280
# features[14:] = the 160- and 320-channel bottleneck stages and the final 1x1 conv.
FINE_TUNE_FROM = 14


def build_mobilenet_v2(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> TransferClassifier:
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
    features = mobilenet_v2(weights=weights).features
    return TransferClassifier(
        architecture="mobilenet_v2",
        backbone=features,
        backbone_channels=MOBILENET_V2_CHANNELS,
        last_block=list(features[FINE_TUNE_FROM:]),
        backbone_gradcam_layer=features[-1],
        normalization=IMAGENET_NORMALIZATION,
        input_size=cfg.input_size,
        additional_conv_layers=cfg.additional_conv_layers,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        num_classes=num_classes,
        freeze_backbone_bn=use_pretrained,
    )
