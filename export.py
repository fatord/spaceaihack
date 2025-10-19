from __future__ import annotations

import os
import tempfile
from typing import Dict

import pandas as pd


def export_trajectory(result: Dict[str, object], filename: str) -> None:
    """write csv: t_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,energy_J,status (atomic)"""
    t = result["t"]
    r = result["r"]
    v = result["v"]
    e = result.get("energy")

    df = pd.DataFrame(
        {
            "t_s": t,
            "x_m": r[:, 0],
            "y_m": r[:, 1],
            "z_m": r[:, 2],
            "vx_m_s": v[:, 0],
            "vy_m_s": v[:, 1],
            "vz_m_s": v[:, 2],
            "energy_J": e if e is not None else None,
            "status": "ok",
        }
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(filename), newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    os.replace(tmp_path, filename)


class CSVAppender:
    """simple append csv for live sessions"""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._f = open(filename, "w", newline="")
        header = (
            "t_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,energy_J,status\n"
        )
        self._f.write(header)
        self._f.flush()

    def append(self, t: float, r, v, energy: float, status: str = "ok") -> None:
        line = f"{t:.6f},{r[0]:.9e},{r[1]:.9e},{r[2]:.9e},{v[0]:.9e},{v[1]:.9e},{v[2]:.9e},{energy:.9e},{status}\n"
        self._f.write(line)
        # flush frequently to make rows visible during live run
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()
