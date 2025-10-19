from __future__ import annotations

import math
import numpy as np

# baseline orbit si units
G = 6.67430e-11
M_SUN = 1.98847e30
M_EARTH = 5.9722e24
AU_M = 149_597_870_700.0

_A = AU_M
_W = math.sqrt(G * M_SUN / (_A ** 3))


def sun_pos(t: float) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0], dtype=float)


def sun_vel(t: float) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0], dtype=float)


def earth_pos(t: float) -> np.ndarray:
    c = math.cos(_W * t)
    s = math.sin(_W * t)
    return np.array([_A * c, _A * s, 0.0], dtype=float)


def earth_vel(t: float) -> np.ndarray:
    c = math.cos(_W * t)
    s = math.sin(_W * t)
    aw = _A * _W
    return np.array([-aw * s, aw * c, 0.0], dtype=float)


def masses() -> dict:
    return {"sun": M_SUN, "earth": M_EARTH}

