"""ResNet comparison model.

The report does not state the depth; ResNet50 is the standard choice in
Keras/TensorFlow transfer-learning studies [STD].
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from skincancer.config import ModelConfig
from skincancer.models.base import TransferClassifier
from skincancer.preprocessing.transforms import IMAGENET_NORMALIZATION

RESNET50_CHANNELS = 2048


def build_resnet50(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> TransferClassifier:
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
    net = resnet50(weights=weights)
    backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
    return TransferClassifier(
        architecture="resnet50",
        backbone=backbone,
        backbone_channels=RESNET50_CHANNELS,
        last_block=[net.layer4],
        backbone_gradcam_layer=net.layer4,
        normalization=IMAGENET_NORMALIZATION,
        input_size=cfg.input_size,
        additional_conv_layers=cfg.additional_conv_layers,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        num_classes=num_classes,
        freeze_backbone_bn=use_pretrained,
    )
