"""Centralised, typed project configuration.

All tunable values live in ``configs/default.yaml``. This module defines the
schema (dataclasses with documented defaults) and loads/validates the YAML.
Every default below is annotated with its evidence class from
``RECONSTRUCTION_SPECIFICATION.md``: [DOC] report text, [FIG] report figure or
screenshot, [SRC] source paper, [STD] standard practice.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

CLASS_NAMES: tuple[str, str] = ("Benign", "Malignant")  # index 0, 1 [DOC]
TRAINABLE_MODES = ("new_layers", "last_block", "all")
# Validation quantity driving checkpoint selection, LR reduction and early stopping -> optimisation direction.
MONITORS = {"val_loss": "min", "val_roc_auc": "max", "val_balanced_accuracy": "max"}


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


@dataclass
class DataConfig:
    images_dir: str = "data/HAM10000/images"
    metadata_csv: str = "data/HAM10000/HAM10000_metadata.csv"
    splits_csv: str = "data/splits/ham10000_lesion_split.csv"
    validation_report: str = "results/data_validation.json"
    # 20% test, then 20% of the remainder for validation -> 64/16/20 overall [FIG]
    test_fraction: float = 0.20
    val_fraction: float = 0.20
    # Binary mapping verified against report Figs. 2-3 [DATA]
    malignant_dx: list[str] = field(default_factory=lambda: ["mel", "bcc", "akiec"])
    benign_dx: list[str] = field(default_factory=lambda: ["nv", "bkl", "df", "vasc"])


@dataclass
class ContrastStretchConfig:
    enabled: bool = True  # [DOC]
    lower_percentile: float = 2.0  # [STD]
    upper_percentile: float = 98.0  # [STD]
    per_channel: bool = False  # global stretch preserves lesion hue [STD]


@dataclass
class GaussianNoiseConfig:
    p: float = 0.25  # [STD]
    variance: float = 0.1  # [DOC], on [0, 1] intensities


@dataclass
class AugmentationConfig:
    horizontal_flip_p: float = 0.5  # [DOC] flip, [STD] probability
    rotation_angles: list[float] = field(default_factory=lambda: [0.0, 15.0, -15.0, 45.0, -45.0])  # [SRC]
    gaussian_noise: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)


@dataclass
class PhaseConfig:
    name: str
    epochs: int
    learning_rate: float
    trainable: str  # one of TRAINABLE_MODES


@dataclass
class ModelConfig:
    architecture: str
    display_name: str
    description: str = ""
    input_size: int = 224
    pretrained: bool = True
    additional_conv_layers: bool = False
    dense_units: int = 256
    dropout: float = 0.0
    phases: list[PhaseConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    batch_size: int = 32  # [FIG]
    num_workers: int = 4
    imbalance_strategy: str = "oversample"  # oversample | class_weight | none
    # Validation ROC-AUC is prevalence-independent; validation loss is distorted when
    # training batches are class-balanced but validation keeps the natural prevalence [STD].
    monitor: str = "val_roc_auc"
    early_stopping_patience: int = 4
    lr_plateau_patience: int = 2
    lr_plateau_factor: float = 0.5
    min_lr: float = 1.0e-7
    log_every_n_batches: int = 50
    device: str = "auto"  # auto | mps | cuda | cpu


@dataclass
class ResourceConfig:
    """Host-friendly limits (defaults tuned for a fanless 8 GB MacBook Air)."""

    nice: int = 10  # lower CPU priority so the Mac stays responsive
    torch_threads: int = 2  # CPU threads in the training process
    mps_memory_fraction: float = 0.55  # of Metal's recommended working set (~3.2 GB on 8 GB M1)
    cooldown_seconds_per_batch: float = 0.35  # GPU duty-cycle throttle at nominal temperature
    fair_cooldown_multiplier: float = 2.0  # longer cooldown while macOS reports "fair"
    thermal_guard: bool = True  # pause at "serious" thermal state until it cools
    max_thermal_pause_seconds: float = 900.0


@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"


@dataclass
class WeightsConfig:
    """Published trained checkpoints, fetched when ``checkpoints/<model>/best.pt`` is missing."""

    release_url: str = ""  # base URL; each asset is f"{release_url}/{model_name}.pt"
    sha256: dict[str, str] = field(default_factory=dict)  # model name -> expected SHA-256 (hex)


@dataclass
class ProjectConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    contrast_stretch: ContrastStretchConfig = field(default_factory=ContrastStretchConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    primary_model: str = "modified_densenet201"
    models: dict[str, ModelConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def resolve(path: str | Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def model(self, name: str) -> ModelConfig:
        try:
            return self.models[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown model '{name}'. Available: {sorted(self.models)}") from exc

    def checkpoint_dir(self, model_name: str) -> Path:
        return self.resolve(self.paths.checkpoints_dir) / model_name

    def results_dir(self, model_name: str | None = None) -> Path:
        base = self.resolve(self.paths.results_dir)
        return base / model_name if model_name else base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- loading
def _build(cls: type, data: Any, where: str) -> Any:
    """Recursively build dataclass ``cls`` from plain YAML data with type checks."""
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        origin = get_origin(hint)
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{where}.{name}")
        elif origin is list and get_args(hint) and is_dataclass(get_args(hint)[0]):
            kwargs[name] = [_build(get_args(hint)[0], v, f"{where}.{name}[{i}]") for i, v in enumerate(value)]
        elif origin is dict and len(get_args(hint)) == 2 and is_dataclass(get_args(hint)[1]):
            kwargs[name] = {k: _build(get_args(hint)[1], v, f"{where}.{name}.{k}") for k, v in value.items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate_config(cfg: ProjectConfig) -> None:
    """Check cross-field constraints; raise ``ConfigError`` on problems."""
    d = cfg.data
    if not (0 < d.test_fraction < 1 and 0 < d.val_fraction < 1):
        raise ConfigError("data.test_fraction and data.val_fraction must be in (0, 1)")
    overlap = set(d.malignant_dx) & set(d.benign_dx)
    if overlap:
        raise ConfigError(f"diagnoses mapped to both classes: {sorted(overlap)}")
    cs = cfg.contrast_stretch
    if not 0 <= cs.lower_percentile < cs.upper_percentile <= 100:
        raise ConfigError("contrast_stretch percentiles must satisfy 0 <= lower < upper <= 100")
    a = cfg.augmentation
    if not 0 <= a.horizontal_flip_p <= 1 or not 0 <= a.gaussian_noise.p <= 1:
        raise ConfigError("augmentation probabilities must be within [0, 1]")
    if a.gaussian_noise.variance < 0:
        raise ConfigError("gaussian_noise.variance must be >= 0")
    if cfg.training.imbalance_strategy not in {"oversample", "class_weight", "none"}:
        raise ConfigError("training.imbalance_strategy must be oversample | class_weight | none")
    r = cfg.resources
    if not 0 < r.mps_memory_fraction <= 1 or r.cooldown_seconds_per_batch < 0 or r.torch_threads < 0:
        raise ConfigError("resources: mps_memory_fraction in (0, 1], cooldown >= 0, torch_threads >= 0 required")
    if cfg.training.monitor not in MONITORS:
        raise ConfigError(f"training.monitor must be one of {sorted(MONITORS)}")
    if cfg.primary_model not in cfg.models:
        raise ConfigError(f"primary_model '{cfg.primary_model}' is not defined under models")
    for name, m in cfg.models.items():
        if not m.phases:
            raise ConfigError(f"models.{name}: at least one training phase is required")
        for p in m.phases:
            if p.trainable not in TRAINABLE_MODES:
                raise ConfigError(f"models.{name}.phases.{p.name}: trainable must be one of {TRAINABLE_MODES}")
            if p.epochs < 1 or p.learning_rate <= 0:
                raise ConfigError(f"models.{name}.phases.{p.name}: epochs >= 1 and learning_rate > 0 required")
        if m.input_size < 32:
            raise ConfigError(f"models.{name}.input_size must be >= 32")
    w = cfg.weights
    if w.release_url and not w.release_url.startswith("https://"):
        raise ConfigError("weights.release_url must be an https:// URL")
    for name, digest in w.sha256.items():
        if name not in cfg.models:
            raise ConfigError(f"weights.sha256.{name}: unknown model")
        if not (isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())):
            raise ConfigError(f"weights.sha256.{name}: expected a 64-character hex SHA-256 digest")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    """Load the project configuration from YAML (defaults to ``configs/default.yaml``)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = _build(ProjectConfig, raw, "config")
    validate_config(cfg)
    return cfg


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """Rebuild a ``ModelConfig`` from the dict stored inside a checkpoint."""
    return _build(ModelConfig, data, "checkpoint.model_config")


def preprocessing_from_dict(data: dict[str, Any]) -> ContrastStretchConfig:
    return _build(ContrastStretchConfig, data, "checkpoint.contrast_stretch")


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
