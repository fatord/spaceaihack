import numpy as np
from gravity import apply_gravity
from thrust import apply_thrust

def objective(thrust_direction, planetary_data, initial_position, initial_velocity, target_position, time_steps, thrust_magnitude, spacecraft_mass):
    thrust_direction = np.array(thrust_direction, dtype=float)
    position = np.array(initial_position, dtype=float)
    velocity = np.array(initial_velocity, dtype=float)
    fuel_used = 0

    for _ in range(len(time_steps)):
        velocity = apply_gravity(position, velocity, planetary_data, spacecraft_mass, time_step=0.1)
        position, velocity = apply_thrust(position, velocity, thrust_direction, thrust_magnitude, time_step=0.1, spacecraft_mass=spacecraft_mass)
        fuel_used += thrust_magnitude * 0.1

    distance_to_target = np.linalg.norm(position - target_position)
    return distance_to_target + fuel_used
