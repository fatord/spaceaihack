# Spacecraft Trajectory Simulator

Spacecraft trajectory sim using Flask with interactive and non-interactive features.
Has CSV and also PNG outputs.

## Requirements

- Python 3.10+
- Packages: `numpy`, `scipy`, `matplotlib`, `pandas`, `requests`, `flask`

## Setup

Use a virtual environment.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy scipy matplotlib pandas requests flask
```

## Run (Flask app)

```
python main.py
```

Open http://127.0.0.1:5000 and start a simulation.

- Simulation runs in a background thread.
- Outputs saved under `static/output/<job_id>/`:
  - `trajectory.csv`
  - `trajectory.png`
- Run interactive simulation.
  - Adjust speeds as needed, sun/earth frames.

## Configuration

Defaults are in `config.py`
