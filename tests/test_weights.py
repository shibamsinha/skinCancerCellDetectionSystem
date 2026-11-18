"""Published-weights download: verification, failure handling, configuration (network is mocked)."""

import hashlib

import pytest
import requests

from skincancer.config import ConfigError, load_config
from skincancer.inference import ModelUnavailableError, WeightsDownloadError, ensure_weights, weights_available
from skincancer.inference import weights as weights_mod

PAYLOAD = b"fake checkpoint bytes" * 1000
GOOD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cfg(tmp_path, sha=GOOD_SHA, url="https://example.org/releases/download/w1"):
    cfg = load_config(overrides={  # overrides are deep-merged, so validation sees googlenet's digest
        "paths": {"checkpoints_dir": str(tmp_path / "ckpt"), "results_dir": str(tmp_path / "res")},
        "weights": {"release_url": url, "sha256": {"googlenet": sha}},
    })
    cfg.weights.sha256 = {"googlenet": sha}  # only googlenet is published in these tests
    return cfg


def _mock_get(monkeypatch, body=PAYLOAD, status=200, calls=None):
    def fake_get(url, stream, timeout):
        if calls is not None:
            calls.append(url)
        return _FakeResponse(body, status)
    monkeypatch.setattr(requests, "get", fake_get)


def test_download_is_verified_and_saved(tmp_path, monkeypatch):
    cfg, calls = _cfg(tmp_path), []
    _mock_get(monkeypatch, calls=calls)
    path = ensure_weights(cfg, "googlenet")
    assert path.read_bytes() == PAYLOAD
    assert calls == ["https://example.org/releases/download/w1/googlenet.pt"]
    # second call uses the local file, no new request
    assert ensure_weights(cfg, "googlenet") == path and len(calls) == 1
    assert not list(path.parent.glob("*.download"))


def test_checksum_mismatch_leaves_nothing_behind(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _mock_get(monkeypatch, body=b"tampered" * 100)
    with pytest.raises(WeightsDownloadError, match="Checksum mismatch"):
        ensure_weights(cfg, "googlenet")
    assert not any((tmp_path / "ckpt" / "googlenet").iterdir())


def test_http_error_is_reported(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _mock_get(monkeypatch, status=404)
    with pytest.raises(WeightsDownloadError, match="Could not download"):
        ensure_weights(cfg, "googlenet")


def test_oversized_download_is_refused(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(weights_mod, "MAX_DOWNLOAD_BYTES", 100)
    _mock_get(monkeypatch)
    with pytest.raises(WeightsDownloadError, match="larger than"):
        ensure_weights(cfg, "googlenet")


def test_local_weights_win_and_missing_weights_raise(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    local = cfg.checkpoint_dir("googlenet") / "best.pt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"locally trained")
    _mock_get(monkeypatch, calls=(calls := []))
    assert ensure_weights(cfg, "googlenet") == local and calls == []
    # resnet50 has no digest in this config and no local file
    assert not weights_available(cfg, "resnet50")
    with pytest.raises(ModelUnavailableError, match="Train it"):
        ensure_weights(cfg, "resnet50")


def test_weights_config_validation(tmp_path):
    with pytest.raises(ConfigError, match="https"):
        _cfg(tmp_path, url="http://example.org/w1")
    with pytest.raises(ConfigError, match="SHA-256"):
        _cfg(tmp_path, sha="abc")
    with pytest.raises(ConfigError, match="unknown model"):
        load_config(overrides={"weights": {"sha256": {"nope": GOOD_SHA}}})


def test_default_config_publishes_all_models():
    cfg = load_config()
    assert cfg.weights.release_url.startswith("https://github.com/")
    assert set(cfg.weights.sha256) == set(cfg.models)
