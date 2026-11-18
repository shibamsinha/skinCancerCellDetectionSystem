"""Model registry: map ``architecture`` names in the config to builder functions."""

from __future__ import annotations

from collections.abc import Callable

from skincancer.config import ModelConfig
from skincancer.models.base import LesionClassifier
from skincancer.models.googlenet import build_googlenet
from skincancer.models.mobilenet import build_mobilenet_v2
from skincancer.models.modified_densenet201 import build_densenet201, build_modified_densenet201
from skincancer.models.resnet import build_resnet50
from skincancer.models.xception import build_xception

Builder = Callable[..., LesionClassifier]

MODEL_BUILDERS: dict[str, Builder] = {
    "modified_densenet201": build_modified_densenet201,
    "densenet201": build_densenet201,
    "mobilenet_v2": build_mobilenet_v2,
    "resnet50": build_resnet50,
    "xception": build_xception,
    "googlenet": build_googlenet,
}


def build_model(cfg: ModelConfig, pretrained: bool | None = None, num_classes: int = 2) -> LesionClassifier:
    """Instantiate the architecture described by ``cfg``.

    Args:
        pretrained: override ``cfg.pretrained``. Inference passes ``False``
            because trained weights are loaded from a checkpoint.
    """
    try:
        builder = MODEL_BUILDERS[cfg.architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture '{cfg.architecture}'. Known: {sorted(MODEL_BUILDERS)}") from exc
    if cfg.architecture == "googlenet":
        return builder(cfg, num_classes=num_classes)
    return builder(cfg, pretrained=pretrained, num_classes=num_classes)
