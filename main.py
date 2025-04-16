# main.py
from flask import Flask, render_template, send_file, redirect, url_for
from simulation import run_simulation
from visualization import plot_trajectory
from export import export_trajectory
import os

app = Flask(__name__)
OUTPUT_CSV = "trajectory_output.csv"
OUTPUT_PLOT = "static/trajectory_plot.png"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run")
def run():
    try:
        trajectory, planet_positions = run_simulation()
        plot_trajectory(trajectory, planet_positions, OUTPUT_PLOT)
        export_trajectory(trajectory, OUTPUT_CSV)
        return redirect(url_for("success"))
    except Exception as e:
        return f"<h2>Error: {e}</h2>"

@app.route("/success")
def success():
    return render_template("success.html", csv_path=OUTPUT_CSV, image_path=OUTPUT_PLOT)

@app.route("/download")
def download():
    return send_file(OUTPUT_CSV, as_attachment=True)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
