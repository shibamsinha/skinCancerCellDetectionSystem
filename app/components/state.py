"""Cached access to configuration, trained models and evaluation artefacts.

* Models are cached with ``st.cache_resource`` keyed by checkpoint modification
  time, so retraining a model is picked up without restarting the app.
* JSON artefacts are cached with ``st.cache_data`` keyed the same way.
* Nothing here loads the dataset; inference needs only the checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from skincancer.config import ProjectConfig, load_config
from skincancer.evaluation.style import MODEL_ORDER
from skincancer.inference import Predictor, ensure_weights, weights_available, weights_local


@st.cache_resource(show_spinner=False)
def get_config() -> ProjectConfig:
    return load_config()


@dataclass(frozen=True)
class ModelStatus:
    name: str
    display_name: str
    description: str
    architecture: str
    input_size: int
    pretrained: bool
    trained: bool
    evaluated: bool
    checkpoint: Path
    is_primary: bool
    training_state: str  # "completed" | "training" | "none"

    @property
    def chip(self) -> tuple[str, str]:
        """(label, style) for status chips."""
        if self.evaluated and self.training_state != "training":
            return "Trained & evaluated", "ok"
        if self.training_state == "training":
            return "Training in progress", "warn"
        if self.trained:
            return "Trained, not evaluated", "primary"
        return "Not trained", "off"


def model_statuses() -> list[ModelStatus]:
    cfg = get_config()
    names = [n for n in MODEL_ORDER if n in cfg.models] + [n for n in cfg.models if n not in MODEL_ORDER]
    out = []
    for n in names:
        m = cfg.models[n]
        ckpt = cfg.checkpoint_dir(n) / "best.pt"
        trained = weights_available(cfg, n)  # local, or published and downloaded on first use
        status = read_json(cfg.results_dir(n) / "status.json") or {}
        out.append(ModelStatus(
            name=n, display_name=m.display_name, description=m.description, architecture=m.architecture,
            input_size=m.input_size, pretrained=m.pretrained, trained=trained,
            evaluated=(cfg.results_dir(n) / "metrics.json").exists(), checkpoint=ckpt,
            is_primary=n == cfg.primary_model,
            training_state=status.get("state", "completed" if trained else "none"),
        ))
    return out


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# At most two models in memory: small hosts (e.g. Streamlit Community Cloud) have limited RAM.
@st.cache_resource(show_spinner=False, max_entries=2)
def _load_predictor(checkpoint: str, _mtime_key: float) -> Predictor:
    predictor = Predictor(checkpoint, device="auto")
    predictor.warmup()  # compile GPU kernels once, so reported inference times are steady-state
    return predictor


def get_predictor(model_name: str) -> Predictor:
    """Load (or reuse) the trained predictor.

    Downloads and verifies the published weights first if they are not on disk.
    Raises ``ModelUnavailableError`` (or its subclass ``WeightsDownloadError``).
    """
    ckpt = ensure_weights(get_config(), model_name)
    return _load_predictor(str(ckpt), _mtime(ckpt))


def needs_download(model_name: str) -> bool:
    """True if the model's weights still have to be fetched (first use on a fresh install)."""
    return not weights_local(get_config(), model_name)


@st.cache_data(show_spinner=False)
def _read_json(path: str, _mtime_key: float) -> Any:
    return json.loads(Path(path).read_text())


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _read_json(str(path), _mtime(path))


def results_summary() -> dict[str, Any] | None:
    return read_json(get_config().results_dir() / "summary.json")


def model_metrics(model_name: str) -> dict[str, Any] | None:
    return read_json(get_config().results_dir(model_name) / "metrics.json")


def model_history(model_name: str) -> list[dict[str, Any]] | None:
    return read_json(get_config().results_dir(model_name) / "history.json")


def training_summary(model_name: str) -> dict[str, Any] | None:
    return read_json(get_config().results_dir(model_name) / "training_summary.json")


def data_report() -> dict[str, Any] | None:
    cfg = get_config()
    return read_json(cfg.resolve(cfg.data.validation_report))
