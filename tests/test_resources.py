import pytest

from skincancer.config import ConfigError, load_config
from skincancer.utils import resources
from skincancer.utils.resources import ThermalGovernor


@pytest.fixture()
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(resources.time, "sleep", lambda s: calls.append(s))
    return calls


def test_default_limits_are_host_friendly():
    r = load_config().resources
    assert r.nice > 0 and r.cooldown_seconds_per_batch > 0 and r.thermal_guard
    assert 0 < r.mps_memory_fraction < 1
    assert load_config().training.num_workers <= 2


def test_invalid_resource_config_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"resources": {"mps_memory_fraction": 1.5}})


def test_cooldown_after_each_batch_at_nominal(monkeypatch, sleeps):
    monkeypatch.setattr(resources, "read_thermal_state", lambda: 0)
    gov = ThermalGovernor(cooldown_seconds=0.3, check_every_n_batches=1)
    for _ in range(3):
        gov.after_batch()
    assert sleeps == [0.3, 0.3, 0.3]


def test_longer_cooldown_when_fair(monkeypatch, sleeps):
    monkeypatch.setattr(resources, "read_thermal_state", lambda: 1)
    gov = ThermalGovernor(cooldown_seconds=0.3, fair_multiplier=2.0, check_every_n_batches=1)
    gov.after_batch()
    assert sleeps == [pytest.approx(0.6)]


def test_pauses_when_serious_until_cool(monkeypatch, sleeps):
    states = iter([2, 2, 1])  # serious at check, still serious after 30 s, then fair
    monkeypatch.setattr(resources, "read_thermal_state", lambda: next(states))
    gov = ThermalGovernor(cooldown_seconds=0.3, check_every_n_batches=1)
    gov.after_batch()
    assert sleeps[:2] == [30, 30]  # paused twice
    assert sleeps[-1] == pytest.approx(0.6)  # then the 'fair' cooldown


def test_disabled_governor_never_sleeps(sleeps):
    gov = ThermalGovernor(enabled=False)
    gov.after_batch()
    assert sleeps == []
