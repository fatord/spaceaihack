# utils
from __future__ import annotations

import math
import os
from typing import Iterable, Optional

import pandas as pd

from config import DEFAULTS


def _default_bodies_df() -> pd.DataFrame:
    """tiny sun + earth set (si units)"""
    AU_M = 149_597_870_700.0
    SUN_MASS_KG = 1.98847e30
    SUN_RADIUS_M = 6.9634e8
    EARTH_MASS_KG = 5.9722e24
    EARTH_RADIUS_M = 6.371e6
    EARTH_SPEED_M_S = 29_780.0

    data = [
        {
            "name": "Sun",
            "mass_kg": SUN_MASS_KG,
            "radius_m": SUN_RADIUS_M,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        {
            "name": "Earth",
            "mass_kg": EARTH_MASS_KG,
            "radius_m": EARTH_RADIUS_M,
            "x": AU_M,
            "y": 0.0,
            "z": 0.0,
            "vx": 0.0,
            "vy": EARTH_SPEED_M_S,
            "vz": 0.0,
        },
    ]
    return pd.DataFrame(data)


def load_planetary_data(selected_bodies: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """load small in-memory data (sun, earth), si + finite"""
    df = _default_bodies_df()

    # filter to select  bodies
    if selected_bodies is None:
        selected_bodies = DEFAULTS.bodies
    selected_set = set(selected_bodies)
    df = df[df["name"].isin(selected_set)].copy()

    # drop invalid rows
    numeric_cols = ["mass_kg", "radius_m", "x", "y", "z", "vx", "vy", "vz"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # ok
    def _is_finite_row(row) -> bool:
        return all(math.isfinite(float(row[c])) for c in numeric_cols)

    df = df[df.apply(_is_finite_row, axis=1)].reset_index(drop=True)
    return df


def get_asteroid_data():
    """placeholder: no remote calls"""
    return []
