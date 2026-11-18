"""GoogleNet (Inception v1 variant) reconstructed exactly from the report's notebook screenshots.

Keras original (report pages 22-23):

    input 64x64x3
    Conv2D(64, 7x7, strides 2, relu, same) -> MaxPool(3x3, strides 2, same)
    Conv2D(64, 1x1, relu, same) -> Conv2D(192, 3x3, relu, same) -> MaxPool(3x3, 2, same)
    inception_module([64, 96, 128, 16, 32, 32])      -> 256 channels
    inception_module([128, 128, 192, 32, 96, 64])    -> 480 channels
    MaxPool(3x3, 2, same)                            -> 4x4x480
    Flatten -> Dense(512, relu) -> Dropout(0.5) -> Dense(NUM_CLASSES, softmax)
    Adam, categorical cross-entropy, 10 epochs, batch size 32

Trained from scratch (no pretrained weights), with no BatchNorm, exactly as
in the screenshot. Keras ``same`` padding is reproduced with symmetric
PyTorch padding, which gives identical output shapes.
"""

from __future__ import annotations

import torch
from torch import nn

from skincancer.config import ModelConfig
from skincancer.models.base import LesionClassifier
from skincancer.preprocessing.transforms import UNIT_NORMALIZATION


def _conv(in_ch: int, out_ch: int, k: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2), nn.ReLU(inplace=False))


class InceptionModule(nn.Module):
    """``inception_module(x, filters)`` from the notebook; ``filters`` = [f0..f5]."""

    def __init__(self, in_ch: int, filters: list[int]) -> None:
        super().__init__()
        f0, f1, f2, f3, f4, f5 = filters
        self.branch_1x1 = _conv(in_ch, f0, 1)
        self.branch_3x3 = nn.Sequential(_conv(in_ch, f1, 1), _conv(f1, f2, 3))
        self.branch_5x5 = nn.Sequential(_conv(in_ch, f3, 1), _conv(f3, f4, 5))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), _conv(in_ch, f5, 1))
        self.out_channels = f0 + f2 + f4 + f5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x), self.branch_pool(x)], dim=1)


class ReportGoogLeNet(LesionClassifier):
    architecture = "googlenet"

    def __init__(self, input_size: int = 64, dense_units: int = 512, dropout: float = 0.5, num_classes: int = 2) -> None:
        super().__init__()
        if input_size % 32 != 0:
            raise ValueError("ReportGoogLeNet expects an input size divisible by 32 (the notebook used 64)")
        self.input_size = input_size
        self.normalization = UNIT_NORMALIZATION
        self.stem = nn.Sequential(
            _conv(3, 64, 7, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            _conv(64, 64, 1),
            _conv(64, 192, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception_1 = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.inception_2 = InceptionModule(self.inception_1.out_channels, [128, 128, 192, 32, 96, 64])
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        spatial = input_size // 16
        flat = self.inception_2.out_channels * spatial * spatial  # 480 * 4 * 4 = 7680 at 64 px
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, dense_units),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception_2(self.inception_1(x))
        return self.classifier(self.pool(x))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.inception_2  # last convolutional block (8x8 at 64 px)

    def set_trainable(self, mode: str) -> None:
        if mode != "all":
            raise ValueError("ReportGoogLeNet is trained from scratch; only trainable='all' is valid")
        for p in self.parameters():
            p.requires_grad = True


def build_googlenet(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> ReportGoogLeNet:
    if cfg.pretrained or pretrained:
        raise ValueError("The report's GoogleNet was trained from scratch; set pretrained: false")
    return ReportGoogLeNet(cfg.input_size, cfg.dense_units, cfg.dropout, num_classes)
