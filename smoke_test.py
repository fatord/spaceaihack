from __future__ import annotations

import os

import numpy as np

from config import DEFAULTS
from export import export_trajectory
from simulation import run_simulation
from utils import load_planetary_data
from visualization import plot_trajectory


def main():
    # for quick run (sanity check pretty much)
    result, pdat = run_simulation(duration_s=3600.0, dt_s=10.0)
    assert np.isfinite(result["r"]).all()
    assert np.isfinite(result["v"]).all()
    assert np.isfinite(result["energy"]).all()

    job_dir = os.path.join(DEFAULTS.output_root, "smoke")
    os.makedirs(job_dir, exist_ok=True)
    csv = os.path.join(job_dir, "trajectory.csv")
    png = os.path.join(job_dir, "trajectory.png")

    export_trajectory(result, csv)
    plot_trajectory(result["r"], pdat, png)
    assert os.path.exists(csv), "CSV not written"
    assert os.path.exists(png), "PNG not written"
    print("Smoke test: OK")


if __name__ == "__main__":
    main()

