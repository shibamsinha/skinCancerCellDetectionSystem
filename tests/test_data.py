import numpy as np
import pandas as pd
import pytest

from skincancer.config import DataConfig, load_config
from skincancer.data import (
    LeakageError,
    MetadataError,
    load_metadata,
    make_lesion_splits,
    validate_images,
    verify_no_leakage,
)
from skincancer.data.dataset import class_weights


def test_binary_mapping(synthetic_data):
    root, _ = synthetic_data
    df = load_metadata(DataConfig(), root / "metadata.csv")
    for dx, expected in {"mel": 1, "bcc": 1, "akiec": 1, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}.items():
        assert set(df.loc[df.dx == dx, "label"]) == {expected}
    assert set(df["label_name"]) == {"Benign", "Malignant"}


def test_unknown_diagnosis_rejected(tmp_path):
    pd.DataFrame([{"lesion_id": "L1", "image_id": "I1", "dx": "scc", "dx_type": "histo", "age": 1,
                   "sex": "male", "localization": "x"}]).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(MetadataError):
        load_metadata(DataConfig(), tmp_path / "m.csv")


def test_missing_metadata_file_gives_helpful_error(tmp_path):
    with pytest.raises(MetadataError, match="download_ham10000"):
        load_metadata(DataConfig(), tmp_path / "missing.csv")


def test_validation_detects_missing_and_corrupt(synthetic_data):
    root, _ = synthetic_data
    df = load_metadata(DataConfig(), root / "metadata.csv")
    (root / "images" / f"{df.image_id[0]}.jpg").unlink()
    (root / "images" / f"{df.image_id[1]}.jpg").write_bytes(b"\xff\xd8\xff garbage")
    valid, report = validate_images(df, root / "images", expected_size=None)
    assert report.missing == [df.image_id[0]]
    assert [c["image_id"] for c in report.corrupted] == [df.image_id[1]]
    assert len(valid) == len(df) - 2 and not report.ok


def test_lesion_split_has_no_leakage_and_is_deterministic(synthetic_data):
    root, _ = synthetic_data
    df = load_metadata(DataConfig(), root / "metadata.csv")
    a = make_lesion_splits(df, 0.2, 0.2, seed=42)
    b = make_lesion_splits(df, 0.2, 0.2, seed=42)
    assert a.equals(b)
    assert set(a["split"]) == {"train", "val", "test"}
    assert len(a) == len(df)
    verify_no_leakage(a)
    frac_test = (a["split"] == "test").mean()
    assert 0.1 < frac_test < 0.3


def test_leakage_detector():
    bad = pd.DataFrame({"image_id": ["a", "b"], "lesion_id": ["L", "L"], "dx": ["nv", "nv"],
                        "label": [0, 0], "split": ["train", "test"]})
    with pytest.raises(LeakageError):
        verify_no_leakage(bad)


def test_class_weights_balance_classes():
    labels = np.array([0] * 80 + [1] * 20)
    w = class_weights(labels).numpy()
    assert w[1] / w[0] == pytest.approx(4.0)
    assert (w[labels].sum() / len(labels)) == pytest.approx(1.0)


def test_real_ham10000_mapping_matches_report_figures():
    """Report Fig. 3: ~1,230 male vs ~725 female malignant cases (skips if data absent)."""
    cfg = load_config()
    path = cfg.resolve(cfg.data.metadata_csv)
    if not path.exists():
        pytest.skip("HAM10000 metadata not downloaded")
    df = load_metadata(cfg.data, path)
    mal = df[df.label == 1]
    assert len(df) == 10015 and len(mal) == 1954
    assert mal.sex.value_counts().to_dict() == {"male": 1227, "female": 727}


def test_saved_real_split_is_leakage_free():
    cfg = load_config()
    path = cfg.resolve(cfg.data.splits_csv)
    if not path.exists():
        pytest.skip("split not created yet")
    splits = pd.read_csv(path)
    verify_no_leakage(splits)
    counts = splits["split"].value_counts(normalize=True)
    assert counts["test"] == pytest.approx(0.20, abs=0.02)
    assert counts["val"] == pytest.approx(0.16, abs=0.02)
