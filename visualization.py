from __future__ import annotations

import matplotlib

# non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory_r: np.ndarray, planetary_data, save_path: str) -> None:
    """save simple 3d png (si units)"""
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajectory_r[:, 0], trajectory_r[:, 1], trajectory_r[:, 2], label="Spacecraft", color="tab:blue", lw=1.2)

    # plot a few bodies to avoid clutter
    max_bodies = 4
    count = 0
    for _, planet in planetary_data.iterrows():
        if count >= max_bodies:
            break
        ax.scatter(float(planet.get("x", 0.0)), float(planet.get("y", 0.0)), float(planet.get("z", 0.0)),
                   label=str(planet.get("name", "Body")), s=20, color="tab:red")
        count += 1

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Spacecraft Trajectory")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

    # scale for it to be readable
    all_points = np.vstack([trajectory_r, planetary_data[["x", "y", "z"]].to_numpy(dtype=float)])
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    for i, set_lim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        pad = 0.05 * (maxs[i] - mins[i] if maxs[i] > mins[i] else 1.0)
        set_lim((mins[i] - pad, maxs[i] + pad))

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
