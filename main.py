from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Dict

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for, stream_with_context

from config import DEFAULTS
from export import export_trajectory
from simulation import run_simulation
from visualization import plot_trajectory
from live_session import LiveConfig, LiveSession


app = Flask(__name__)


# j*b registry
JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
MAX_CONCURRENT_JOBS = 2

# live sessions registry
SESSIONS: Dict[str, LiveSession] = {}
SESSIONS_LOCK = threading.Lock()


def _static_job_paths(job_id: str) -> dict:
    root = os.path.join(DEFAULTS.output_root, job_id)
    csv = os.path.join(root, "trajectory.csv")
    png = os.path.join(root, "trajectory.png")
    return {"root": root, "csv": csv, "png": png}


def _set_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)


def _progress_cb_factory(job_id: str):
    def _cb(step, total_steps, t, r, v, e):
        pct = int(100 * step / max(total_steps, 1))
        _set_job(job_id, progress=pct, t=t, r=r.tolist(), v=v.tolist(), e=float(e))
    return _cb


def _worker(job_id: str, duration_s: float, dt_s: float, bodies):
    try:
        _set_job(job_id, status="running", started_at=time.time())
        progress_cb = _progress_cb_factory(job_id)
        result, pdat = run_simulation(duration_s=duration_s, dt_s=dt_s, bodies=bodies, progress_cb=progress_cb)

        # save the artifacts
        paths = _static_job_paths(job_id)
        os.makedirs(paths["root"], exist_ok=True)

        # CSV write handled inside export_trajectory
        export_trajectory(result, paths["csv"])

        # png write save to temp and then replace
        tmp_png = os.path.join(paths["root"], f"trajectory_{uuid.uuid4().hex[:6]}.tmp.png")
        plot_trajectory(result["r"], pdat, tmp_png)
        os.replace(tmp_png, paths["png"])

        import numpy as _np
        radii = _np.linalg.norm(result["r"], axis=1)
        summary = {
            "total_steps": int(result["t"].shape[0] - 1),
            "elapsed_s": float(result["t"][-1]),
            "final_r_m": result["r"][-1].tolist(),
            "final_v_m_s": result["v"][-1].tolist(),
            "min_radius_m": float(radii.min()),
            "max_radius_m": float(radii.max()),
        }

        _set_job(job_id, status="complete", progress=100, finished_at=time.time(), paths=paths, summary=summary)
    except Exception as e:
        _set_job(job_id, status="error", error=str(e), finished_at=time.time())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    # cap
    with JOBS_LOCK:
        running = sum(1 for j in JOBS.values() if j.get("status") == "running")
        if running >= MAX_CONCURRENT_JOBS:
            return render_template("error.html", message="Too many concurrent jobs. Please try again later."), 429

    try:
        duration_s = float(request.form.get("duration_s", DEFAULTS.duration_s))
        dt_s = float(request.form.get("dt_s", DEFAULTS.dt_s))
        bodies = request.form.getlist("bodies") or list(DEFAULTS.bodies)
    except Exception:
        return render_template("error.html", message="Invalid parameters."), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "pending",
            "progress": 0,
            "params": {"duration_s": duration_s, "dt_s": dt_s, "bodies": bodies},
            "created_at": time.time(),
        }

    t = threading.Thread(target=_worker, args=(job_id, duration_s, dt_s, bodies), daemon=True)
    t.start()
    return redirect(url_for("success", job_id=job_id))


@app.route("/success/<job_id>")
def success(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return render_template("error.html", message="Unknown job_id."), 404
    return render_template("success.html", job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job_id"}), 404

    data = {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "t": job.get("t"),
        "r": job.get("r"),
        "v": job.get("v"),
        "e": job.get("e"),
        "summary": job.get("summary"),
    }
    if job.get("paths"):
        data.update(
            {
                "csv_url": url_for("static", filename=f"output/{job_id}/trajectory.csv"),
                "png_url": url_for("static", filename=f"output/{job_id}/trajectory.png"),
            }
        )
    return jsonify(data)


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "unknown job_id"}), 404

    def event_stream():
        last_progress = -1
        while True:
            job = JOBS.get(job_id, {})
            status = job.get("status", "unknown")
            progress = int(job.get("progress", 0))
            # does not work but oh well
            if status == "complete":
                progress = 100
            payload = {"status": status, "progress": progress, "t": job.get("t")}
            if job.get("paths"):
                payload.update(
                    {
                        "csv_url": url_for("static", filename=f"output/{job_id}/trajectory.csv"),
                        "png_url": url_for("static", filename=f"output/{job_id}/trajectory.png"),
                    }
                )

            if progress != last_progress or status in {"complete", "error"}:
                yield f"data: {json.dumps(payload)}\n\n"
                last_progress = progress

            if status in {"complete", "error"}:
                break
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/result/<job_id>")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job_id"}), 404
    if job.get("status") != "complete":
        return jsonify({"error": "job not complete", "status": job.get("status")}), 409

    return jsonify(
        {
            "job_id": job_id,
            "summary": job.get("summary", {}),
            "csv_url": url_for("static", filename=f"output/{job_id}/trajectory.csv"),
            "png_url": url_for("static", filename=f"output/{job_id}/trajectory.png"),
        }
    )

@app.route("/start", methods=["GET", "POST"])
def start_live():
    try:
        if request.method == "POST":
            duration_s = float(request.form.get("duration_s", DEFAULTS.duration_s))
            dt_s = float(request.form.get("dt_s", DEFAULTS.dt_s))
            initial_speed = float(request.form.get("speed", 10.0))
            frame = request.form.get("frame", "sun")
        else:
            # GET with defaults for convenience
            duration_s = DEFAULTS.duration_s
            dt_s = DEFAULTS.dt_s
            initial_speed = 10.0
            frame = "sun"
    except Exception:
        return render_template("error.html", message="Invalid parameters."), 400

    session_id = uuid.uuid4().hex[:12]
    cfg = LiveConfig(
        duration_s=duration_s,
        dt_s=dt_s,
        softening_m=DEFAULTS.softening_m,
        render_hz=15.0,
        initial_speed=initial_speed,
        frame=frame,
    )
    sess = LiveSession(session_id, cfg, DEFAULTS.output_root)
    with SESSIONS_LOCK:
        SESSIONS[session_id] = sess
    sess.start()
    # redirect
    return redirect(url_for("live_dashboard", session_id=session_id))


@app.route("/live/<session_id>")
def live_dashboard(session_id: str):
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            return render_template("error.html", message="Unknown session_id."), 404
    return render_template("live.html", session_id=session_id)


@app.route("/stream/live/<session_id>")
def stream_live(session_id: str):
    with SESSIONS_LOCK:
        sess = SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "unknown session_id"}), 404

    @stream_with_context
    def event_stream():
        # send one snapshot immediately
        try:
            first = sess.get_latest_snapshot()
            yield f"data: {json.dumps(first)}\n\n"
        except Exception:
            pass
        while True:
            snap = sess.get_snapshot(timeout=1.0)
            if not snap:
                # q
                time.sleep(0.05)
                continue
            yield f"data: {json.dumps(snap)}\n\n"
            if snap.get("status") in {"done", "error"}:
                break

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(event_stream(), headers=headers, mimetype="text/event-stream")


@app.route("/control/<session_id>", methods=["POST"])
def control(session_id: str):
    with SESSIONS_LOCK:
        sess = SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "unknown session_id"}), 404

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    action = payload.get("action")
    if action not in {"pause", "resume", "set_speed", "step", "reset", "stop"}:
        return jsonify({"error": "unknown action"}), 400

    sess.enqueue(payload)
    return jsonify({"ok": True})


@app.route("/state/<session_id>")
def state(session_id: str):
    with SESSIONS_LOCK:
        sess = SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "unknown session_id"}), 404
    # always return a latest snapshot for polling clients
    snap = sess.get_latest_snapshot()
    return jsonify(snap)


@app.route("/stop/<session_id>", methods=["POST"])
def stop(session_id: str):
    with SESSIONS_LOCK:
        sess = SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "unknown session_id"}), 404
    sess.enqueue({"action": "stop"})
    return jsonify({"ok": True})


if __name__ == "__main__":
    os.makedirs(DEFAULTS.output_root, exist_ok=True)
    # no reloader; enable threads for  background loop
    app.run(debug=True, use_reloader=False, threaded=True)