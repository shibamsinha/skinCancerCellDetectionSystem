"""Phase-wise training loop with checkpointing, resume and early stopping.

Training follows the report's "retraining the Deep Learning model through
multiple phases" [DOC], reconstructed as standard transfer learning [STD]:

1. ``transfer_learning``: backbone frozen; train the additional conv layers + head.
2. ``fine_tuning``: also unfreeze the backbone's last stage, at a lower learning rate.

The optimiser is Adam and the loss is cross-entropy on the 2-way softmax [FIG].
The best weights by the monitored validation metric (default: ROC-AUC) are
kept, and restored before each new phase and at the end of training (Keras
``ModelCheckpoint(save_best_only=True)``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from skincancer.config import CLASS_NAMES, MONITORS, ProjectConfig
from skincancer.data.dataset import DataLoaders, build_dataloaders
from skincancer.data.pipeline import load_prepared_splits
from skincancer.data.splits import summarize_splits
from skincancer.evaluation.metrics import binary_metrics
from skincancer.models import LesionClassifier, build_model
from skincancer.preprocessing.augmentation import TrainAugmentation
from skincancer.preprocessing.transforms import ImageTransform
from skincancer.training.callbacks import BestTracker, EarlyStopping
from skincancer.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from skincancer.utils.device import resolve_device
from skincancer.utils.logging import configure_logging, get_logger
from skincancer.utils.resources import ThermalGovernor, apply_process_limits
from skincancer.utils.seed import seed_everything

log = get_logger(__name__)


def build_transforms(cfg: ProjectConfig, model: LesionClassifier) -> tuple[ImageTransform, ImageTransform]:
    """(train_transform with augmentation, eval_transform without)."""
    train_tf = ImageTransform(model.input_size, model.normalization, cfg.contrast_stretch, TrainAugmentation(cfg.augmentation))
    eval_tf = ImageTransform(model.input_size, model.normalization, cfg.contrast_stretch, None)
    return train_tf, eval_tf


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader, device: torch.device, governor: ThermalGovernor | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (y_true, prob_malignant, mean cross-entropy loss) over a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    ys, probs, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y))
        n += len(y)
        probs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
        ys.append(y.cpu().numpy())
        if governor is not None:
            governor.after_batch()
    return np.concatenate(ys), np.concatenate(probs), loss_sum / max(n, 1)


class Trainer:
    def __init__(self, cfg: ProjectConfig, model_name: str, num_workers: int | None = None) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.model_cfg = cfg.model(model_name)
        self.device = resolve_device(cfg.training.device)
        self.ckpt_dir = cfg.checkpoint_dir(model_name)
        self.out_dir = cfg.results_dir(model_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(log_file=self.out_dir / "train.log")

        res = cfg.resources
        apply_process_limits(res.nice, res.torch_threads, res.mps_memory_fraction, self.device)
        self.governor = ThermalGovernor(
            cooldown_seconds=res.cooldown_seconds_per_batch,
            fair_multiplier=res.fair_cooldown_multiplier,
            max_pause_seconds=res.max_thermal_pause_seconds,
            enabled=res.thermal_guard or res.cooldown_seconds_per_batch > 0,
        )
        if not res.thermal_guard:
            self.governor.pause_at_state = 99  # cooldown only, never pause
        self.generator = seed_everything(cfg.seed)
        self.model = build_model(self.model_cfg).to(self.device)
        self.splits = load_prepared_splits(cfg)
        train_tf, eval_tf = build_transforms(cfg, self.model)
        self.loaders: DataLoaders = build_dataloaders(
            self.splits,
            cfg.resolve(cfg.data.images_dir),
            train_tf,
            eval_tf,
            cfg.training.batch_size,
            cfg.training.num_workers if num_workers is None else num_workers,
            cfg.training.imbalance_strategy,
            self.generator,
        )
        weights = self.loaders.loss_weights
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device) if weights is not None else None)
        self.history: list[dict[str, Any]] = []
        self.monitor = cfg.training.monitor
        self.mode = MONITORS[self.monitor]
        self.best = BestTracker(mode=self.mode)

    # ------------------------------------------------------------- paths
    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.pt"

    # ------------------------------------------------------------ helpers
    def _make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.trainable_parameters(), lr=lr)

    def _make_scheduler(self, optimizer: torch.optim.Optimizer):
        t = self.cfg.training
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.mode, factor=t.lr_plateau_factor, patience=t.lr_plateau_patience, min_lr=t.min_lr
        )

    def _cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def _save_best(self, phase_name: str, global_epoch: int, val: dict[str, float]) -> None:
        save_checkpoint(
            self.best_path,
            {
                "model_name": self.model_name,
                "architecture": self.model_cfg.architecture,
                "model_config": asdict(self.model_cfg),
                "contrast_stretch": asdict(self.cfg.contrast_stretch),
                "class_names": list(CLASS_NAMES),
                "input_size": self.model.input_size,
                "normalization": {"mean": list(self.model.normalization.mean), "std": list(self.model.normalization.std)},
                "gradcam_target": self.model.gradcam_target_name,
                "state_dict": self._cpu_state_dict(),
                "epoch": global_epoch,
                "phase": phase_name,
                "val_metrics": {k: float(v) for k, v in val.items()},
                "monitor": self.monitor,
                "seed": self.cfg.seed,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "torch_version": str(torch.__version__),
                "device": str(self.device),
            },
        )

    def _save_last(self, state: dict[str, Any]) -> None:
        save_checkpoint(self.last_path, {**state, "model_state": self._cpu_state_dict(), "rng": capture_rng_state()})

    def _write_history(self) -> None:
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def _write_status(self, state: str, **extra: Any) -> None:
        """Small JSON heartbeat read by the web app (training / completed)."""
        payload = {"state": state, "epochs_done": len(self.history),
                   "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
        (self.out_dir / "status.json").write_text(json.dumps(payload, indent=2))

    def _restore_best_weights(self) -> None:
        if self.best_path.exists():
            ckpt = load_checkpoint(self.best_path)
            self.model.load_state_dict(ckpt["state_dict"])
            log.info("Restored best weights (epoch %s, phase %s)", ckpt["epoch"], ckpt["phase"])

    # -------------------------------------------------------------- epoch
    def _train_epoch(self, optimizer: torch.optim.Optimizer, tag: str) -> dict[str, float]:
        self.model.train()
        loader = self.loaders.train
        loss_sum, correct, n = 0.0, 0, 0
        t0 = time.time()
        every = self.cfg.training.log_every_n_batches
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {i}: {loss.item()}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            self.governor.after_batch()
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)
            if every and i % every == 0:
                log.info("%s batch %d/%d loss=%.4f acc=%.4f (%.1f img/s)",
                         tag, i, len(loader), loss_sum / n, correct / n, n / (time.time() - t0))
        return {"loss": loss_sum / n, "accuracy": correct / n}

    # ---------------------------------------------------------------- fit
    def fit(self, resume: bool = False) -> list[dict[str, Any]]:
        phases = self.model_cfg.phases
        start_phase, start_epoch = 0, 0
        resume_state: dict[str, Any] | None = None

        if resume and self.last_path.exists():
            resume_state = load_checkpoint(self.last_path)
            if resume_state.get("completed"):
                log.info("Training for %s already completed; nothing to resume.", self.model_name)
                self.history = resume_state["history"]
                return self.history
            self.model.load_state_dict(resume_state["model_state"])
            self.history = resume_state["history"]
            self.best.load_state_dict(resume_state["best"])
            start_phase, start_epoch = resume_state["phase_index"], resume_state["next_epoch"]
            restore_rng_state(resume_state["rng"])
            if start_epoch == 0:
                # Interrupted exactly at a phase boundary: the new phase starts
                # from the best weights with a fresh optimiser, as in a normal run.
                self._restore_best_weights()
                resume_state = None
            log.info("Resuming %s at phase %d (%s), epoch %d", self.model_name, start_phase,
                     phases[start_phase].name, start_epoch + 1)
        elif resume:
            log.warning("--resume requested but %s does not exist; starting fresh", self.last_path)
        else:
            # Fresh run: keep one backup of an earlier run instead of silently
            # mixing its best.pt into this run's model selection.
            for stale in (self.best_path, self.last_path):
                if stale.exists():
                    backup = stale.with_name(stale.stem + ".previous.pt")
                    stale.replace(backup)
                    log.warning("Existing %s moved to %s (fresh run)", stale.name, backup.name)

        split_summary = summarize_splits(self.splits)
        log.info("Model %s on %s | train=%d val=%d test=%d | imbalance=%s",
                 self.model_name, self.device, split_summary["train"]["images"], split_summary["val"]["images"],
                 split_summary["test"]["images"], self.cfg.training.imbalance_strategy)
        run_start = time.time()
        self._write_status("training", phase=phases[start_phase].name, phase_epoch=start_epoch,
                           phase_epochs=phases[start_phase].epochs)

        for p_idx in range(start_phase, len(phases)):
            phase = phases[p_idx]
            self.model.set_trainable(phase.trainable)
            params = self.model.count_parameters()
            optimizer = self._make_optimizer(phase.learning_rate)
            scheduler = self._make_scheduler(optimizer)
            stopper = EarlyStopping(self.cfg.training.early_stopping_patience, mode=self.mode)
            first_epoch = 0
            if resume_state is not None and p_idx == start_phase:
                optimizer.load_state_dict(resume_state["optimizer"])
                scheduler.load_state_dict(resume_state["scheduler"])
                stopper.load_state_dict(resume_state["early_stopping"])
                first_epoch = start_epoch
            log.info("Phase %d/%d '%s': trainable=%s lr=%.1e epochs=%d | params trainable %.2fM / %.2fM",
                     p_idx + 1, len(phases), phase.name, phase.trainable, phase.learning_rate, phase.epochs,
                     params["trainable"] / 1e6, params["total"] / 1e6)

            stopped = False
            for epoch in range(first_epoch, phase.epochs):
                t0 = time.time()
                tag = f"[{self.model_name}|{phase.name}|ep {epoch + 1}/{phase.epochs}]"
                train_stats = self._train_epoch(optimizer, tag)
                y_true, prob, val_loss = predict_loader(self.model, self.loaders.val, self.device, self.governor)
                val = binary_metrics(y_true, prob)
                lr = optimizer.param_groups[0]["lr"]
                global_epoch = len(self.history) + 1
                record = {
                    "epoch": global_epoch,
                    "phase": phase.name,
                    "phase_epoch": epoch + 1,
                    "lr": lr,
                    "train_loss": train_stats["loss"],
                    "train_accuracy": train_stats["accuracy"],
                    "val_loss": val_loss,
                    "val_accuracy": val["accuracy"],
                    "val_balanced_accuracy": val["balanced_accuracy"],
                    "val_sensitivity": val["sensitivity"],
                    "val_specificity": val["specificity"],
                    "val_roc_auc": val["roc_auc"],
                    "seconds": round(time.time() - t0, 1),
                }
                self.history.append(record)
                score = record[self.monitor]
                finite = math.isfinite(score)  # ROC-AUC is undefined if validation has one class
                if not finite:
                    log.warning("%s is undefined this epoch; it is ignored for model selection", self.monitor)
                # The first epoch always produces a checkpoint, so best.pt exists after any run.
                improved = (finite and self.best.is_improvement(score)) or self.best.epoch < 0
                if improved:
                    self.best.update(score if finite else self.best.best, global_epoch, phase.name)
                    self._save_best(phase.name, global_epoch, {"val_loss": val_loss, **val})
                if finite:
                    scheduler.step(score)
                stopped = stopper.step(score) if finite else False
                log.info("%s train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f bal_acc=%.4f auc=%.4f "
                         "sens=%.3f spec=%.3f | %.0fs%s",
                         tag, train_stats["loss"], train_stats["accuracy"], val_loss, val["accuracy"],
                         val["balanced_accuracy"], val["roc_auc"], val["sensitivity"], val["specificity"],
                         record["seconds"], " *best*" if improved else "")
                self._write_history()
                self._write_status("training", phase=phase.name, phase_epoch=epoch + 1, phase_epochs=phase.epochs)
                next_phase, next_epoch = (p_idx, epoch + 1) if (epoch + 1 < phase.epochs and not stopped) else (p_idx + 1, 0)
                self._save_last({
                    "phase_index": next_phase,
                    "next_epoch": next_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "early_stopping": stopper.state_dict(),
                    "best": self.best.state_dict(),
                    "history": self.history,
                    "completed": next_phase >= len(phases),
                })
                if stopped:
                    log.info("Early stopping in phase '%s' after %d epochs", phase.name, epoch + 1)
                    break
            resume_state = None
            self._restore_best_weights()  # next phase starts from the best weights so far
            if self.device.type == "mps":
                torch.mps.empty_cache()

        summary = {
            "model_name": self.model_name,
            "device": str(self.device),
            "epochs_run": len(self.history),
            "best_epoch": self.best.epoch,
            "best_phase": self.best.phase,
            "monitor": self.monitor,
            "best_monitor_value": self.best.best,
            "training_seconds_this_session": round(time.time() - run_start, 1),
            "splits": split_summary,
            "imbalance_strategy": self.cfg.training.imbalance_strategy,
            "model_config": asdict(self.model_cfg),
            "augmentation": asdict(self.cfg.augmentation),
            "contrast_stretch": asdict(self.cfg.contrast_stretch),
            "batch_size": self.cfg.training.batch_size,
            "seed": self.cfg.seed,
        }
        (self.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        self._write_status("completed", best_epoch=self.best.epoch)
        log.info("Training complete for %s. Best %s=%.4f at epoch %d (%s).",
                 self.model_name, self.monitor, self.best.best, self.best.epoch, self.best.phase)
        return self.history
