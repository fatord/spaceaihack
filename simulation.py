#simulation
import numpy as np
from gravity import apply_gravity

def simulate_trajectory(time_steps, initial_position, velocity, planetary_data, spacecraft_mass):
    positions = []
    position = initial_position.copy()
    for t in time_steps:
        velocity = apply_gravity(position, velocity, planetary_data, spacecraft_mass, time_step=0.1)
        position += velocity * 0.1
        positions.append(position.copy())
	print(f"Step {t}: pos = {position}, vel = {velocity}")
    return np.array(positions)

from utils import load_planetary_data  # or wherever you're getting data from

def run_simulation():
    planetary_data = load_planetary_data()
    initial_position = np.array([1e11, 0, 0], dtype=float)
    initial_velocity = np.array([0, 30000, 0], dtype=float)
    spacecraft_mass = 500  # in kg
    time_steps = np.linspace(0, 1000, 100)  # simulate for 100 steps
    trajectory = simulate_trajectory(time_steps, initial_position, initial_velocity, planetary_data, spacecraft_mass)
    return trajectory, planetary_data