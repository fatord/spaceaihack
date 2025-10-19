"""simple config (si units)"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    # integration
    dt_s: float = 10.0  # [s]
    duration_s: float = 24 * 3600.0  # [s] 1 day
    softening_m: float = 1.0e6  # [m]
    progress_every_steps: int = 100  # throttle progress updates

    # default bodies
    bodies: tuple[str, ...] = ("Sun", "Earth")

    # output
    output_root: str = "static/output"


DEFAULTS = Defaults()
