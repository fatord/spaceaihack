from __future__ import annotations

import math
import numpy as np

from config import DEFAULTS

# g const
G = 6.67430e-11


def gravitational_acceleration(position: np.ndarray, planetary_data) -> np.ndarray:
    """grav accel at pos from all bodies"""
    ax = np.zeros(3, dtype=float)
    eps2 = float(DEFAULTS.softening_m) ** 2

    for _, planet in planetary_data.iterrows():
        m = float(planet.get("mass_kg", 0.0))
        px = float(planet.get("x", 0.0))
        py = float(planet.get("y", 0.0))
        pz = float(planet.get("z", 0.0))
        if not (math.isfinite(m) and math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            continue

        rvec = np.array([px, py, pz], dtype=float) - position
        r2 = float(np.dot(rvec, rvec)) + eps2
        inv_r3 = 1.0 / (r2 * math.sqrt(r2))
        ax += G * m * inv_r3 * rvec

    return ax


def apply_gravity(position, velocity, planetary_data, spacecraft_mass, time_step):
    """compat shim: update v with grav accel (mass-independent)"""
    a = gravitational_acceleration(np.asarray(position, dtype=float), planetary_data)
    v = np.asarray(velocity, dtype=float) + a * float(time_step)
    return v
