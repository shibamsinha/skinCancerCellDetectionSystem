"""Modified DenseNet201, the primary model of the original project.

Architecture (report Fig. 5; source paper Table 2):

    DenseNet201 (ImageNet-pretrained)
      conv0 7x7/2 -> pool0 3x3/2
      dense block 1 [1x1, 3x3] x 6   -> transition 1 (1x1 conv, 2x2 avg-pool/2)
      dense block 2 [1x1, 3x3] x 12  -> transition 2
      dense block 3 [1x1, 3x3] x 48  -> transition 3
      dense block 4 [1x1, 3x3] x 32  -> norm5 -> ReLU          (1920 x 7 x 7 at 224 px)
    Additional layers
      CONV1_1  3x1 conv, 128 filters, ReLU                     (128 x 7 x 7)
      CONV1_2  1x3 conv, 128 filters, ReLU                     (128 x 7 x 7)
      CONV2    3x3 conv,  64 filters, ReLU                     ( 64 x 7 x 7)  <- Grad-CAM target
    Classification head
      GAP -> BatchNorm -> Dense(256, ReLU) -> Dense(2) -> softmax

The backbone is torchvision's DenseNet201, which is topologically identical
to ``keras.applications.DenseNet201`` and uses the same input normalisation.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import DenseNet201_Weights, densenet201

from skincancer.config import ModelConfig
from skincancer.models.base import TransferClassifier
from skincancer.preprocessing.transforms import IMAGENET_NORMALIZATION

DENSENET201_CHANNELS = 1920


def build_densenet201_backbone(pretrained: bool) -> tuple[nn.Sequential, list[nn.Module], nn.Module]:
    """Return (backbone, last_block, final_activation).

    The backbone ends with the ReLU that torchvision applies after ``norm5``
    (Keras' ``include_top=False`` output is also post-ReLU).
    """
    weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
    features = densenet201(weights=weights).features
    final_act = nn.ReLU(inplace=False)
    backbone = nn.Sequential(features, final_act)
    last_block = [features.denseblock4, features.norm5]
    return backbone, last_block, final_act


def build_modified_densenet201(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> TransferClassifier:
    if not cfg.additional_conv_layers:
        raise ValueError(
            "modified_densenet201 requires additional_conv_layers: true; "
            "use the 'densenet201' architecture for the unmodified ablation."
        )
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    backbone, last_block, final_act = build_densenet201_backbone(use_pretrained)
    return TransferClassifier(
        architecture="modified_densenet201",
        backbone=backbone,
        backbone_channels=DENSENET201_CHANNELS,
        last_block=last_block,
        backbone_gradcam_layer=final_act,
        normalization=IMAGENET_NORMALIZATION,
        input_size=cfg.input_size,
        additional_conv_layers=True,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        num_classes=num_classes,
        freeze_backbone_bn=use_pretrained,
    )


def build_densenet201(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> TransferClassifier:
    """Unmodified DenseNet201 + the same head, for the ablation comparison."""
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    backbone, last_block, final_act = build_densenet201_backbone(use_pretrained)
    return TransferClassifier(
        architecture="densenet201",
        backbone=backbone,
        backbone_channels=DENSENET201_CHANNELS,
        last_block=last_block,
        backbone_gradcam_layer=final_act,
        normalization=IMAGENET_NORMALIZATION,
        input_size=cfg.input_size,
        additional_conv_layers=cfg.additional_conv_layers,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        num_classes=num_classes,
        freeze_backbone_bn=use_pretrained,
    )
