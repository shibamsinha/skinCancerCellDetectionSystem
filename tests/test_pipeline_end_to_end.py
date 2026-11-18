"""Integration tests on a tiny synthetic dataset (CPU, no DataLoader workers)."""

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from skincancer.data.pipeline import prepare_dataset
from skincancer.evaluation.evaluate import evaluate_model
from skincancer.explainability import GradCAM
from skincancer.inference import ModelUnavailableError, Predictor
from skincancer.models import build_model
from skincancer.training.checkpointing import load_checkpoint
from skincancer.training.trainer import Trainer


def _shorten(cfg, name, epochs=1, size=None):
    for p in cfg.model(name).phases:
        p.epochs = epochs
    if size:
        cfg.model(name).input_size = size


def test_prepare_dataset_writes_split_and_report(synthetic_cfg):
    splits = prepare_dataset(synthetic_cfg)
    assert set(splits["split"]) == {"train", "val", "test"}
    report = json.loads(synthetic_cfg.resolve(synthetic_cfg.data.validation_report).read_text())
    assert report["validation"]["ok"] and report["splits"]["test"]["images"] > 0


def test_googlenet_train_evaluate_predict(synthetic_cfg):
    cfg = synthetic_cfg
    _shorten(cfg, "googlenet", epochs=2)
    trainer = Trainer(cfg, "googlenet")
    history = trainer.fit()
    assert len(history) == 2 and {"train_loss", "val_loss", "val_roc_auc"} <= set(history[0])
    ckpt = load_checkpoint(trainer.best_path)
    assert ckpt["model_name"] == "googlenet" and ckpt["gradcam_target"] == "inception_2"
    last = load_checkpoint(trainer.last_path)
    assert last["completed"] is True

    result = evaluate_model(cfg, "googlenet")
    out = cfg.results_dir("googlenet")
    for f in ("metrics.json", "predictions.csv", "confusion_matrix.png", "roc_curve.png", "training_curves.png"):
        assert (out / f).exists(), f
    m = result["metrics"]
    assert m["n"] == (trainer.splits["split"] == "test").sum()
    assert sum(sum(r) for r in result["confusion_matrix"]["matrix"]) == m["n"]
    assert json.loads((cfg.results_dir() / "summary.json").read_text())["models"]["googlenet"]

    pred = Predictor.for_model(cfg, "googlenet", device="cpu")
    buf = io.BytesIO()
    Image.new("RGB", (600, 450), (120, 80, 70)).save(buf, format="JPEG")
    p = pred.predict(buf.getvalue())
    assert p.label in ("Benign", "Malignant")
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert p.confidence == max(p.probabilities.values())
    e = pred.explain(buf.getvalue())
    assert e.original.shape == e.heatmap.shape == e.overlay.shape == (64, 64, 3)
    assert 0.0 <= e.cam.min() and e.cam.max() <= 1.0


def test_modified_densenet_two_phase_training_and_resume(synthetic_cfg):
    cfg = synthetic_cfg
    _shorten(cfg, "modified_densenet201", epochs=1, size=64)
    cfg.model("modified_densenet201").pretrained = False  # keep the test offline and fast
    t = Trainer(cfg, "modified_densenet201")
    hist = t.fit()
    assert [h["phase"] for h in hist] == ["transfer_learning", "fine_tuning"]
    # Resuming a completed run is a no-op that returns the saved history.
    t2 = Trainer(cfg, "modified_densenet201")
    assert len(t2.fit(resume=True)) == 2


def test_resume_mid_phase_continues_from_next_epoch(synthetic_cfg):
    cfg = synthetic_cfg
    _shorten(cfg, "googlenet", epochs=3)
    t = Trainer(cfg, "googlenet")
    # Simulate an interruption after epoch 1 by running a 1-epoch phase, then
    # restoring the configured 3 epochs and resuming.
    cfg.model("googlenet").phases[0].epochs = 1
    t.fit()
    state = load_checkpoint(t.last_path)
    assert state["completed"] is True
    # Pretend the run was interrupted: mark the phase as not finished.
    state["completed"], state["phase_index"], state["next_epoch"] = False, 0, 1
    torch.save(state, t.last_path)
    cfg.model("googlenet").phases[0].epochs = 3
    hist = Trainer(cfg, "googlenet").fit(resume=True)
    assert [h["phase_epoch"] for h in hist] == [1, 2, 3]


def test_predictor_reports_missing_weights(synthetic_cfg):
    with pytest.raises(ModelUnavailableError):
        Predictor.for_model(synthetic_cfg, "xception", device="cpu")


def test_gradcam_on_modified_densenet(project_cfg):
    model = build_model(project_cfg.model("modified_densenet201"), pretrained=False).eval()
    x = torch.rand(1, 3, 224, 224)
    with GradCAM(model, model.gradcam_target_layer) as cam:
        res = cam(x)
        res_other = cam(x, class_index=1 - res.class_index)
    assert res.cam.shape == (224, 224) and res.raw_cam.shape == (7, 7)
    assert res.cam.min() >= 0 and res.cam.max() <= 1
    assert res_other.class_index != res.class_index
    assert np.isclose(res.probabilities.sum(), 1.0, atol=1e-5)
    # Hooks are removed after the context exits.
    assert not model.additional.conv2._forward_hooks
    with pytest.raises(ValueError):
        GradCAM(model, model.gradcam_target_layer)(torch.rand(2, 3, 224, 224))


def test_gradcam_validation_runs(project_cfg):
    from skincancer.explainability.validation import validate_gradcam

    model = build_model(project_cfg.model("modified_densenet201"), pretrained=False).eval()
    images = [torch.rand(1, 3, 64, 64) for _ in range(2)]
    res = validate_gradcam(model, "additional.conv2", images, ["additional", "head"], area_fractions=(0.2,), random_repeats=1)
    assert res.n_images == 2 and set(res.cam_drop_mean) == {"20%"}
    assert -1.0 <= res.randomization_spearman_mean <= 1.0
