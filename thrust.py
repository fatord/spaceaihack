# thrust.py

import numpy as np

# Constants
spacecraft_mass = 500  # kg

def apply_thrust(position, velocity, thrust_direction, thrust_magnitude, time_step):
    """
    Applies thrust to the spacecraft.

    Parameters:
        position (np.array): Current position of the spacecraft.
        velocity (np.array): Current velocity of the spacecraft.
        thrust_direction (np.array): Unit vector in the direction of thrust.
        thrust_magnitude (float): Magnitude of thrust in Newtons.
        time_step (float): Duration of time step in seconds.

    Returns:
        tuple: Updated position and velocity after applying thrust.
    """
    force = thrust_direction * thrust_magnitude
    acceleration = force / spacecraft_mass
    velocity += acceleration * time_step
    position += velocity * time_step
    return position, velocity
