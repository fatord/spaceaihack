# gravity.py
import numpy as np
G = 6.67430e-11  # Gravitational constant

# Applies gravitational acceleration from planetary data to the spacecraft
def apply_gravity(position, velocity, planetary_data, spacecraft_mass, time_step):
    for _, planet in planetary_data.iterrows():
        planet_mass = planet.get('mass', 0) * 1.989e30  # Convert from solar mass to kg
        planet_position = np.array([planet.get('x', 0), planet.get('y', 0), planet.get('z', 0)])
        distance_vector = planet_position - position
        distance = np.linalg.norm(distance_vector)
        if distance > 0:
            gravitational_force = G * spacecraft_mass * planet_mass / distance**2
            gravitational_acceleration = (gravitational_force / spacecraft_mass) * (distance_vector / distance)
            velocity += gravitational_acceleration * time_step
    return velocity
