"""Keep training gentle on the host machine (built for a fanless MacBook Air, 8 GB).

* ``apply_process_limits``: lower CPU priority, cap CPU threads, cap GPU (MPS)
  memory so the process raises an out-of-memory error instead of pushing the
  system into heavy swapping.
* ``ThermalGovernor``: a duty-cycle throttle. After every training batch it
  sleeps for a short cooldown, longer when macOS reports elevated thermal
  state. When the state is ``serious`` or worse, it pauses until the machine
  cools down.

macOS thermal states (NSProcessInfo.thermalState): 0 nominal, 1 fair, 2 serious, 3 critical.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import torch

from skincancer.utils.logging import get_logger

log = get_logger(__name__)

THERMAL_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def apply_process_limits(nice: int, torch_threads: int, mps_memory_fraction: float, device: torch.device) -> None:
    """Apply CPU-priority, thread and GPU-memory limits to the current process.

    DataLoader workers are started afterwards, so they inherit the lower priority.
    """
    if nice > 0:
        try:
            current = os.nice(0)  # os.nice is cumulative; only raise the niceness up to the target
            if current < nice:
                os.nice(nice - current)
        except OSError as exc:  # pragma: no cover - permission/platform dependent
            log.warning("Could not lower process priority: %s", exc)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    if device.type == "mps" and 0 < mps_memory_fraction < 1:
        torch.mps.set_per_process_memory_fraction(mps_memory_fraction)
        cap = mps_memory_fraction * torch.mps.recommended_max_memory() / 1e9
        log.info("Resource limits: nice +%d, %d CPU threads, MPS memory cap %.1f GB", nice, torch_threads, cap)
    else:
        log.info("Resource limits: nice +%d, %d CPU threads", nice, torch_threads)


def read_thermal_state() -> int | None:
    """macOS thermal state 0-3, or ``None`` when it cannot be read (other OSes)."""
    if platform.system() != "Darwin":
        return None
    try:
        from Foundation import NSProcessInfo  # pyobjc

        return int(NSProcessInfo.processInfo().thermalState())
    except Exception:  # noqa: BLE001 - pyobjc missing or API unavailable
        return None


@dataclass
class ThermalGovernor:
    """Per-batch cooldown, adapted to the current macOS thermal state."""

    cooldown_seconds: float = 0.35  # after every batch at nominal temperature
    fair_multiplier: float = 2.0  # longer cooldown while the state is "fair"
    pause_at_state: int = 2  # pause completely at "serious" (2) or worse
    resume_below_state: int = 2  # ...until the state drops below this
    check_every_n_batches: int = 10
    max_pause_seconds: float = 900.0
    enabled: bool = True
    _batches: int = 0
    _state: int | None = None
    paused_seconds: float = 0.0

    def after_batch(self) -> None:
        if not self.enabled:
            return
        if self._batches % self.check_every_n_batches == 0:
            self._state = read_thermal_state()
            if self._state is not None and self._state >= self.pause_at_state:
                self._pause_until_cool()
        self._batches += 1
        delay = self.cooldown_seconds
        if self._state is not None and self._state >= 1:
            delay *= self.fair_multiplier
        if delay > 0:
            time.sleep(delay)

    def _pause_until_cool(self) -> None:
        start = time.time()
        log.warning("Thermal state '%s': pausing training until the Mac cools down",
                    THERMAL_NAMES.get(self._state, self._state))
        while time.time() - start < self.max_pause_seconds:
            time.sleep(30)
            self._state = read_thermal_state()
            if self._state is None or self._state < self.resume_below_state:
                break
        waited = time.time() - start
        self.paused_seconds += waited
        log.info("Resuming after %.0f s pause (thermal state now '%s')", waited,
                 THERMAL_NAMES.get(self._state, self._state))
