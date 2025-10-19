import numpy as np

# simple thrust
spacecraft_mass = 500

def apply_thrust(position, velocity, thrust_direction, thrust_magnitude, time_step):
    """apply thrust dv over one step"""
    force = thrust_direction * thrust_magnitude
    a = force / spacecraft_mass
    velocity += a * time_step
    position += velocity * time_step
    return position, velocity
