import random

import numpy as np
import pytest
import torch

from skincancer.evaluation import binary_metrics, confusion_matrix, per_diagnosis_rates
from skincancer.training.callbacks import EarlyStopping
from skincancer.training.checkpointing import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint


def test_binary_metrics_known_values():
    # TP=40 FN=10 TN=90 FP=10
    y = np.array([1] * 50 + [0] * 100)
    p = np.array([0.9] * 40 + [0.1] * 10 + [0.1] * 90 + [0.9] * 10)
    m = binary_metrics(y, p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (40, 10, 90, 10)
    assert m["accuracy"] == pytest.approx(130 / 150)
    assert m["sensitivity"] == pytest.approx(0.8) and m["recall"] == m["sensitivity"]
    assert m["specificity"] == pytest.approx(0.9)
    assert m["precision"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx(0.85)
    assert m["majority_class_accuracy"] == pytest.approx(100 / 150)
    assert min(m["sensitivity"], m["specificity"]) <= m["accuracy"] <= max(m["sensitivity"], m["specificity"])


def test_perfect_classifier_auc():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0


def test_lesion_level_resampling_keeps_lesions_together():
    from skincancer.evaluation.metrics import _Resampler

    groups = np.array(["a", "a", "a", "b", "c", "c"])
    resample = _Resampler(len(groups), groups, seed=0)
    for _ in range(50):
        idx = resample()
        picked = groups[idx]
        # every sampled lesion contributes all of its images (a multiple of its image count)
        for g, size in (("a", 3), ("b", 1), ("c", 2)):
            assert np.sum(picked == g) % size == 0
        assert len(np.unique(idx[picked == "a"])) in (0, 3)


def test_paired_bootstrap_difference():
    from skincancer.evaluation.metrics import paired_bootstrap_difference

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.clip(y * 0.6 + rng.random(400) * 0.4, 0, 1)  # separates the classes well
    noise = rng.random(400)  # chance level
    groups = np.repeat(np.arange(200), 2)
    res = paired_bootstrap_difference(y, good, noise, groups=groups, n_boot=300, seed=1)
    auc = res["roc_auc"]
    assert auc["difference"] > 0.3 and auc["ci_low"] > 0 and auc["p_value"] < 0.01
    same = paired_bootstrap_difference(y, good, good, n_boot=100)["roc_auc"]
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0 and same["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_difference(y, good, noise[:10])


def test_confusion_matrix_layout():
    cm = confusion_matrix(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]))
    assert cm.tolist() == [[1, 1], [1, 2]]  # rows = truth, cols = prediction


def test_per_diagnosis_rates():
    out = per_diagnosis_rates(np.array(["mel", "mel", "nv"]), np.array([1, 0, 0]))
    assert out["mel"]["predicted_malignant_rate"] == 0.5 and out["nv"]["n"] == 1


def test_checkpoint_roundtrip_is_weights_only_safe(tmp_path):
    payload = {"state_dict": {"w": torch.ones(3)}, "meta": {"epoch": 3, "name": "x", "vals": [1.5, None]},
               "rng": capture_rng_state()}
    save_checkpoint(tmp_path / "c.pt", payload)
    loaded = load_checkpoint(tmp_path / "c.pt")  # torch.load(weights_only=True)
    assert torch.equal(loaded["state_dict"]["w"], torch.ones(3)) and loaded["meta"]["epoch"] == 3


def test_rng_state_restores_exact_sequence():
    state = capture_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng_state(state)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_early_stopping():
    es = EarlyStopping(patience=2)
    assert not es.step(1.0)
    assert not es.step(1.1)
    assert es.step(1.2)


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.80)
    assert not es.step(0.85)  # improvement resets the counter
    assert not es.step(0.84)
    assert es.step(0.83)


def test_best_tracker_modes():
    from skincancer.training.callbacks import BestTracker

    hi = BestTracker(mode="max")
    assert hi.is_improvement(0.5)
    hi.update(0.9, 1, "p")
    assert not hi.is_improvement(0.8) and hi.is_improvement(0.95)
    lo = BestTracker(mode="min")
    lo.update(0.3, 1, "p")
    assert lo.is_improvement(0.2) and not lo.is_improvement(0.4)
