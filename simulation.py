from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from config import DEFAULTS
from gravity import gravitational_acceleration
from utils import load_planetary_data


@dataclass
class SimParams:
    dt_s: float = DEFAULTS.dt_s
    duration_s: float = DEFAULTS.duration_s
    softening_m: float = DEFAULTS.softening_m
    bodies: Iterable[str] = DEFAULTS.bodies
    progress_every_steps: int = DEFAULTS.progress_every_steps


def _energy(state_r: np.ndarray, state_v: np.ndarray, planetary_data) -> float:
    """specific energy j/kg (kin + pot)"""
    # kinetic per unit mass
    ke = 0.5 * float(np.dot(state_v, state_v))
    # potential energy
    pe = 0.0
    for _, p in planetary_data.iterrows():
        m = float(p["mass_kg"])
        r = float(np.linalg.norm(np.array([p["x"], p["y"], p["z"]], dtype=float) - state_r))
        r = max(r, 1.0)  # avoid div by zero
        pe -= 6.67430e-11 * m / r
    return ke + pe


def _rk4_step(r: np.ndarray, v: np.ndarray, dt: float, planetary_data) -> tuple[np.ndarray, np.ndarray]:
    """rk4 step for r, v"""
    a1 = gravitational_acceleration(r, planetary_data)
    k1_r = v
    k1_v = a1

    a2 = gravitational_acceleration(r + 0.5 * dt * k1_r, planetary_data)
    k2_r = v + 0.5 * dt * k1_v
    k2_v = a2

    a3 = gravitational_acceleration(r + 0.5 * dt * k2_r, planetary_data)
    k3_r = v + 0.5 * dt * k2_v
    k3_v = a3

    a4 = gravitational_acceleration(r + dt * k3_r, planetary_data)
    k4_r = v + dt * k3_v
    k4_v = a4

    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next


def simulate_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    planetary_data,
    params: Optional[SimParams] = None,
    progress_cb=None,
):
    """integrate in inertial frame; returns dict t,r,v,energy"""
    if params is None:
        params = SimParams()

    dt = float(params.dt_s)
    steps = int(math.ceil(params.duration_s / dt))

    r = np.array(initial_position, dtype=float)
    v = np.array(initial_velocity, dtype=float)

    t_arr = np.zeros(steps + 1, dtype=float)
    r_arr = np.zeros((steps + 1, 3), dtype=float)
    v_arr = np.zeros((steps + 1, 3), dtype=float)
    e_arr = np.zeros(steps + 1, dtype=float)

    r_arr[0] = r
    v_arr[0] = v
    e_arr[0] = _energy(r, v, planetary_data)

    for i in range(1, steps + 1):
        r, v = _rk4_step(r, v, dt, planetary_data)

        # so that we don't get NaN
        if not np.isfinite(r).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"Non-finite state at step {i}: r={r}, v={v}")

        t_arr[i] = i * dt
        r_arr[i] = r
        v_arr[i] = v
        e_arr[i] = _energy(r, v, planetary_data)

        if progress_cb and (i % params.progress_every_steps == 0):
            try:
                progress_cb(i, steps, t_arr[i], r.copy(), v.copy(), e_arr[i])
            except Exception:
                # never break integrator
                pass

    return {"t": t_arr, "r": r_arr, "v": v_arr, "energy": e_arr}


def run_simulation(
    duration_s: Optional[float] = None,
    dt_s: Optional[float] = None,
    bodies: Optional[Iterable[str]] = None,
    progress_cb=None,
):
    """run default demo; returns (result, bodies df)"""
    params = SimParams(
        dt_s=dt_s if dt_s is not None else DEFAULTS.dt_s,
        duration_s=duration_s if duration_s is not None else DEFAULTS.duration_s,
        bodies=tuple(bodies) if bodies is not None else DEFAULTS.bodies,
    )

    planetary_data = load_planetary_data(selected_bodies=params.bodies)

    # initial state
    initial_position = np.array([1.0e11, 0.0, 0.0], dtype=float)
    initial_velocity = np.array([0.0, 3.0e4, 0.0], dtype=float)

    result = simulate_trajectory(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        planetary_data=planetary_data,
        params=params,
        progress_cb=progress_cb,
    )
    return result, planetary_data
