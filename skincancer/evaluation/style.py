"""Shared chart palette (validated categorical slots on a white surface).

Colours follow the entity, never its rank: every model keeps the same hue in
every chart, in both the static PNGs and the Streamlit app.
"""

from __future__ import annotations

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e4"
REFERENCE_LINE = "#a3a29c"

# Categorical slots 1-6 (validated: CVD dE >= 9.1, normal-vision dE >= 19.6).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

MODEL_ORDER = ["modified_densenet201", "densenet201", "mobilenet_v2", "resnet50", "xception", "googlenet"]
MODEL_COLORS = dict(zip(MODEL_ORDER, SERIES))

TRAIN_COLOR = SERIES[0]
VAL_COLOR = SERIES[1]

# Sequential blue ramp (light -> dark) for confusion-matrix cells.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, SERIES[0])
