"""Xception comparison model.

Uses timm's ``legacy_xception``, a port of the original Keras Xception
ImageNet weights, with Keras' [-1, 1] input scaling.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from skincancer.config import ModelConfig
from skincancer.models.base import TransferClassifier
from skincancer.preprocessing.transforms import INCEPTION_NORMALIZATION

XCEPTION_CHANNELS = 2048
TIMM_NAME = "legacy_xception.tf_in1k"


class _TimmFeatures(nn.Module):
    """Expose a timm model's ``forward_features`` as a plain module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


def build_xception(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> TransferClassifier:
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    net = timm.create_model(TIMM_NAME, pretrained=use_pretrained, num_classes=0, global_pool="")
    net.act4 = nn.ReLU(inplace=False)  # non-inplace so Grad-CAM hooks see clean tensors
    backbone = _TimmFeatures(net)
    exit_flow = [net.block12, net.conv3, net.bn3, net.conv4, net.bn4]
    return TransferClassifier(
        architecture="xception",
        backbone=backbone,
        backbone_channels=XCEPTION_CHANNELS,
        last_block=exit_flow,
        backbone_gradcam_layer=net.act4,
        normalization=INCEPTION_NORMALIZATION,
        input_size=cfg.input_size,
        additional_conv_layers=cfg.additional_conv_layers,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        num_classes=num_classes,
        freeze_backbone_bn=use_pretrained,
    )
