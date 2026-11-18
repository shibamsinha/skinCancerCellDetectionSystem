"""Building blocks shared by all reconstructed architectures.

* ``AdditionalConvLayers``: CONV1_1 (3x1, 128) -> CONV1_2 (1x3, 128) -> CONV2 (3x3, 64),
  the three layers appended after DenseNet201's fourth dense block
  (report Fig. 5 [FIG]; kernel sizes and CONV2 filters from the source paper's Table 2 [SRC]).
* ``ClassificationHead``: GAP -> BatchNorm -> Dense 1 -> Dense 2 (+ softmax)
  (report Fig. 4 [FIG]).
* ``TransferClassifier``: pretrained backbone + optional additional layers + head,
  with phase-wise freezing for transfer learning and fine-tuning.

All classifiers return **logits** of shape ``(N, 2)``. Softmax is applied in
``predict_proba``; training uses ``CrossEntropyLoss``, which equals Keras'
softmax + categorical cross-entropy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from skincancer.preprocessing.transforms import Normalization


class ConvReLU(nn.Module):
    """``Conv2d`` followed by ReLU, the equivalent of Keras ``Conv2D(..., activation='relu')``.

    Wrapping both in one module means a forward hook on it captures the
    post-activation feature maps, which Grad-CAM needs.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], padding: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class AdditionalConvLayers(nn.Module):
    """The three convolutional layers appended after the backbone.

    The 3x3 receptive field is factorised into a 3x1 then a 1x3 convolution
    to reduce computation (report §6 [DOC]). ``same`` padding and stride 1
    keep the spatial size, e.g. 7x7 for a 224x224 input. The ReLU activation
    is reconstructed [STD].
    """

    CONV1_FILTERS = 128  # [DOC]/[SRC]
    CONV2_FILTERS = 64  # [SRC]

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1_1 = ConvReLU(in_channels, self.CONV1_FILTERS, kernel_size=(3, 1), padding=(1, 0))
        self.conv1_2 = ConvReLU(self.CONV1_FILTERS, self.CONV1_FILTERS, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = ConvReLU(self.CONV1_FILTERS, self.CONV2_FILTERS, kernel_size=(3, 3), padding=(1, 1))
        self.out_channels = self.CONV2_FILTERS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1_2(self.conv1_1(x)))


class ClassificationHead(nn.Module):
    """GAP -> BatchNorm -> Dense 1 (ReLU) -> [Dropout] -> Dense 2 (logits for softmax)."""

    def __init__(self, in_channels: int, dense_units: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.dense1 = nn.Linear(in_channels, dense_units)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dense2 = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.gap(x), 1)
        x = self.batch_norm(x)
        x = self.dropout(self.act(self.dense1(x)))
        return self.dense2(x)


class LesionClassifier(nn.Module):
    """Interface shared by every model used by the trainer, Grad-CAM and the app."""

    architecture: str = "base"
    normalization: Normalization
    input_size: int

    @property
    def gradcam_target_layer(self) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def gradcam_target_name(self) -> str:
        target = self.gradcam_target_layer
        for name, module in self.named_modules():
            if module is target:
                return name
        return type(target).__name__

    def set_trainable(self, mode: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax class probabilities, shape ``(N, num_classes)``."""
        return torch.softmax(self(x), dim=1)


def _set_requires_grad(modules: Iterable[nn.Module], flag: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


class TransferClassifier(LesionClassifier):
    """Pretrained backbone + optional additional conv layers + classification head.

    Args:
        backbone: module mapping images to a ``(N, C, h, w)`` feature map.
        backbone_channels: ``C``.
        last_block: backbone sub-modules unfrozen in the fine-tuning phase.
        backbone_gradcam_layer: Grad-CAM target used when there are no additional layers.
        freeze_backbone_bn: keep backbone BatchNorm layers in inference mode
            (Keras transfer-learning convention). Used for pretrained backbones.
    """

    def __init__(
        self,
        architecture: str,
        backbone: nn.Module,
        backbone_channels: int,
        last_block: list[nn.Module],
        backbone_gradcam_layer: nn.Module,
        normalization: Normalization,
        input_size: int,
        additional_conv_layers: bool,
        dense_units: int,
        dropout: float,
        num_classes: int = 2,
        freeze_backbone_bn: bool = True,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = backbone
        self.additional = AdditionalConvLayers(backbone_channels) if additional_conv_layers else None
        head_in = self.additional.out_channels if self.additional is not None else backbone_channels
        self.head = ClassificationHead(head_in, dense_units, num_classes, dropout)
        # These modules are already registered inside ``backbone``. Bypass
        # nn.Module.__setattr__ so they are not registered a second time
        # (which would duplicate their weights in the state_dict).
        object.__setattr__(self, "_last_block", list(last_block))
        object.__setattr__(self, "_backbone_gradcam_layer", backbone_gradcam_layer)
        self.normalization = normalization
        self.input_size = input_size
        self.freeze_backbone_bn = freeze_backbone_bn

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.additional is not None:
            features = self.additional(features)
        return self.head(features)

    # ------------------------------------------------------------- Grad-CAM
    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last convolutional layer before global average pooling."""
        if self.additional is not None:
            return self.additional.conv2
        return self._backbone_gradcam_layer

    # ------------------------------------------------------- phase handling
    def new_layers(self) -> list[nn.Module]:
        return [m for m in (self.additional, self.head) if m is not None]

    def set_trainable(self, mode: str) -> None:
        """``new_layers``: train only additional convs + head (backbone frozen).
        ``last_block``: also fine-tune the backbone's last stage.
        ``all``: train everything."""
        if mode == "all":
            _set_requires_grad([self], True)
            return
        _set_requires_grad([self.backbone], False)
        _set_requires_grad(self.new_layers(), True)
        if mode == "last_block":
            _set_requires_grad(self._last_block, True)
        elif mode != "new_layers":
            raise ValueError(f"Unknown trainable mode '{mode}'")

    def train(self, mode: bool = True) -> "TransferClassifier":
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self
