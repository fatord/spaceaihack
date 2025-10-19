from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
import logging
from typing import Deque, Dict, Optional

import numpy as np

from config import DEFAULTS
from orbits import G, M_EARTH, M_SUN, earth_pos, earth_vel, sun_pos
from export import CSVAppender


def _accel_dynamic(r_sc: np.ndarray, t_sim: float, softening_m: float) -> np.ndarray:
    """grav accel at spacecraft pos (sun + earth)"""
    eps2 = softening_m ** 2

    # sun at origin
    r_sun_sc = -r_sc  # vec from sc to sun
    r2 = float(np.dot(r_sun_sc, r_sun_sc)) + eps2
    a_sun = G * M_SUN * r_sun_sc / (r2 * math.sqrt(r2))

    # earth at time t
    rE = earth_pos(t_sim)
    r_earth_sc = rE - r_sc
    r2e = float(np.dot(r_earth_sc, r_earth_sc)) + eps2
    a_earth = G * M_EARTH * r_earth_sc / (r2e * math.sqrt(r2e))

    return a_sun + a_earth


def _rk4_step_dyn(r: np.ndarray, v: np.ndarray, dt: float, t_sim: float, softening_m: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = _accel_dynamic(r, t_sim, softening_m)
    k1_r = v
    k1_v = a1

    a2 = _accel_dynamic(r + 0.5 * dt * k1_r, t_sim + 0.5 * dt, softening_m)
    k2_r = v + 0.5 * dt * k1_v
    k2_v = a2

    a3 = _accel_dynamic(r + 0.5 * dt * k2_r, t_sim + 0.5 * dt, softening_m)
    k3_r = v + 0.5 * dt * k2_v
    k3_v = a3

    a4 = _accel_dynamic(r + dt * k3_r, t_sim + dt, softening_m)
    k4_r = v + dt * k3_v
    k4_v = a4

    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next


def _energy_dynamic(r: np.ndarray, v: np.ndarray, t_sim: float) -> float:
    ke = 0.5 * float(np.dot(v, v))
    # potential per unit mass: -GM/|r - r_body|
    r_s = sun_pos(t_sim)
    r_e = earth_pos(t_sim)
    pe = -G * M_SUN / max(1.0, float(np.linalg.norm(r - r_s)))
    pe += -G * M_EARTH / max(1.0, float(np.linalg.norm(r - r_e)))
    return ke + pe


def _ang_mom(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.cross(r, v)


@dataclass
class LiveConfig:
    duration_s: float
    dt_s: float
    softening_m: float
    render_hz: float = 15.0
    initial_speed: float = 10.0
    frame: str = "sun"  # or "earth"
    ring_max: int = 2000


class LiveSession:
    def __init__(self, session_id: str, cfg: LiveConfig, output_root: str) -> None:
        self._log = logging.getLogger(f"LiveSession[{session_id}]")
        self.session_id = session_id
        self.cfg = cfg
        self.output_root = output_root
        self.paths = {
            "root": os.path.join(output_root, session_id),
            "csv": os.path.join(output_root, session_id, "trajectory.csv"),
            "png": os.path.join(output_root, session_id, "trajectory.png"),
        }
        os.makedirs(self.paths["root"], exist_ok=True)

        # state
        self.t_sim: float = 0.0
        self.r: np.ndarray
        self.v: np.ndarray
        self._init_state()

        # controls
        self.speed: float = max(0.0, float(cfg.initial_speed))
        self.paused: bool = False
        self.stopped: bool = False
        self.control_q: "queue.Queue[dict]" = queue.Queue()

        # streaming
        self.snapshot_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self.ring: Deque[tuple[float, float]] = deque(maxlen=cfg.ring_max)
        self.ring_geo: Deque[tuple[float, float]] = deque(maxlen=cfg.ring_max)

        # csv appender
        self.csv = CSVAppender(self.paths["csv"])

        # thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

        # full trajectory for final png
        self._traj_r = [self.r.copy()]

    def _init_state(self) -> None:
        # start near earth with small dv
        rE0 = earth_pos(0.0)
        vE0 = earth_vel(0.0)
        self.r = rE0 + np.array([300e3, 0.0, 0.0], dtype=float)
        self.v = vE0 + np.array([0.0, 3200.0, 0.0], dtype=float)
        self.t_sim = 0.0

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, cmd: dict) -> None:
        self._log.info("control %s", cmd)
        self.control_q.put(cmd)

    def stop(self) -> None:
        self.stopped = True

    def get_snapshot(self, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            return self.snapshot_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_snapshot(self) -> dict:
        # build a snapshot from current state (not queued)
        rE = earth_pos(self.t_sim)
        energy = _energy_dynamic(self.r, self.v, self.t_sim)
        h = _ang_mom(self.r, self.v)
        progress = int(math.floor(100.0 * min(self.t_sim, self.cfg.duration_s) / max(1e-9, self.cfg.duration_s)))
        return {
            "session_id": self.session_id,
            "status": "paused" if self.paused else ("done" if self.t_sim >= self.cfg.duration_s else "running"),
            "progress": progress,
            "t_sim": self.t_sim,
            "speed": self.speed,
            "frame": self.cfg.frame,
            "r": self.r.tolist(),
            "v": self.v.tolist(),
            "r_norm": float(np.linalg.norm(self.r)),
            "v_norm": float(np.linalg.norm(self.v)),
            "energy_J": float(energy),
            "h": h.tolist(),
            "ring_xy": list(self.ring),
            "ring_geo_xy": list(self.ring_geo),
            "csv_url": f"/static/output/{self.session_id}/trajectory.csv",
            "png_url": f"/static/output/{self.session_id}/trajectory.png",
        }

    def _emit_snapshot(self, status: str) -> None:
        # update ring buffers
        rE = earth_pos(self.t_sim)
        # sun frame xy
        self.ring.append((float(self.r[0]), float(self.r[1])))
        # earth frame xy
        geo = self.r - rE
        self.ring_geo.append((float(geo[0]), float(geo[1])))

        energy = _energy_dynamic(self.r, self.v, self.t_sim)
        h = _ang_mom(self.r, self.v)
        progress = int(math.floor(100.0 * min(self.t_sim, self.cfg.duration_s) / max(1e-9, self.cfg.duration_s)))

        payload = {
            "session_id": self.session_id,
            "status": status,
            "progress": progress,
            "t_sim": self.t_sim,
            "speed": self.speed,
            "frame": self.cfg.frame,
            "r": self.r.tolist(),
            "v": self.v.tolist(),
            "r_norm": float(np.linalg.norm(self.r)),
            "v_norm": float(np.linalg.norm(self.v)),
            "energy_J": float(energy),
            "h": h.tolist(),
            "ring_xy": list(self.ring),
            "ring_geo_xy": list(self.ring_geo),
            "csv_url": f"/static/output/{self.session_id}/trajectory.csv",
            "png_url": f"/static/output/{self.session_id}/trajectory.png",
        }
        try:
            if self.snapshot_q.full():
                _ = self.snapshot_q.get_nowait()
            self.snapshot_q.put_nowait(payload)
        except queue.Full:
            pass

    def _handle_controls(self) -> None:
        # Drain quickly; last command of each type wins
        last_speed = None
        step_count = 0
        do_reset = False
        while True:
            try:
                cmd = self.control_q.get_nowait()
            except queue.Empty:
                break
            action = cmd.get("action")
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "set_speed":
                try:
                    sp = float(cmd.get("speed", self.speed))
                    last_speed = max(0.0, sp)
                except Exception:
                    pass
            elif action == "step":
                step_count += 1
            elif action == "reset":
                do_reset = True
            elif action == "stop":
                self.stopped = True
            # loggable actions could be added here if we need to

        if last_speed is not None:
            self.speed = last_speed
        return step_count, do_reset

    def _run_loop(self) -> None:
        tick_dt = 1.0 / max(1.0, float(self.cfg.render_hz))
        last = time.perf_counter()
        carry = 0.0

        # initial snapshot
        self._emit_snapshot(status="running")
        self.csv.append(self.t_sim, self.r, self.v, _energy_dynamic(self.r, self.v, self.t_sim), status="ok")

        try:
            while not self.stopped and self.t_sim < self.cfg.duration_s:
                step_count, do_reset = self._handle_controls()
                if do_reset:
                    self._init_state()
                    carry = 0.0

                now = time.perf_counter()
                elapsed = now - last
                if elapsed < tick_dt:
                    time.sleep(tick_dt - elapsed)
                    now = time.perf_counter()
                    elapsed = now - last
                last = now

                if self.paused and step_count == 0:
                    # when paused, still emit to keep ui alive
                    self._emit_snapshot(status="paused")
                    continue

                # how much sim time to advance
                sim_advance = self.speed * elapsed + carry
                n_steps = int(sim_advance // self.cfg.dt_s)
                carry = sim_advance - n_steps * self.cfg.dt_s

                # if paused and step requested, do exactly that
                if self.paused and step_count > 0:
                    n_steps = min(step_count, 100)  # safety cap
                    step_count = 0

                # cap steps per tick
                n_steps = min(n_steps, 200)

                # integrate
                for _ in range(n_steps):
                    self.r, self.v = _rk4_step_dyn(self.r, self.v, self.cfg.dt_s, self.t_sim, self.cfg.softening_m)
                    if not (np.isfinite(self.r).all() and np.isfinite(self.v).all()):
                        raise FloatingPointError("Non-finite state during integration")
                    self.t_sim += self.cfg.dt_s
                    # append every step for simplicity
                    self.csv.append(self.t_sim, self.r, self.v, _energy_dynamic(self.r, self.v, self.t_sim), status="ok")
                    self._traj_r.append(self.r.copy())

                # emit one snapshot per tick
                if self.paused:
                    self._emit_snapshot(status="paused")
                else:
                    self._emit_snapshot(status="running")

            # finalize
            self.t_sim = min(self.t_sim, self.cfg.duration_s)
            # final csv row + final snapshot
            self.csv.append(self.t_sim, self.r, self.v, _energy_dynamic(self.r, self.v, self.t_sim), status="done")
            self._emit_snapshot(status="done")
            # save final png
            try:
                import numpy as _np
                from visualization import plot_trajectory
                tmp_png = os.path.join(self.paths["root"], f"trajectory_{int(time.time())}.tmp.png")
                plot_trajectory(_np.asarray(self._traj_r), _session_planets_df(self.t_sim), tmp_png)
                os.replace(tmp_png, self.paths["png"])
            except Exception:
                pass
        except Exception as e:
            # ERROR path: write an error row and emit error snapshot
            try:
                self.csv.append(self.t_sim, self.r, self.v, _energy_dynamic(self.r, self.v, self.t_sim), status="error")
            except Exception:
                pass
            err_payload = {
                "session_id": self.session_id,
                "status": "error",
                "progress": int(math.floor(100.0 * min(self.t_sim, self.cfg.duration_s) / max(1e-9, self.cfg.duration_s))),
                "t_sim": self.t_sim,
                "message": str(e),
            }
            try:
                if self.snapshot_q.full():
                    _ = self.snapshot_q.get_nowait()
                self.snapshot_q.put_nowait(err_payload)
            except queue.Full:
                pass
        finally:
            try:
                self.csv.close()
            except Exception:
                pass


def _session_planets_df(t: float):
    """tiny df for plotting bodies"""
    import pandas as _pd
    rE = earth_pos(t)
    return _pd.DataFrame([
        {"name": "sun", "x": 0.0, "y": 0.0, "z": 0.0},
        {"name": "earth", "x": float(rE[0]), "y": float(rE[1]), "z": float(rE[2])},
    ])
