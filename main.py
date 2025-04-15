import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import requests
import pandas as pd

def load_planetary_data():
    url = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.txt"
    planetary_data = pd.read_csv(url)
    return planetary_data

spacecraft_model = "Orion Multi-Purpose Crew Vehicle"
spacecraft_mass = 500
thrust_magnitude = 10
initial_position = np.array([0.0, 0.0, 0.0])
initial_velocity = np.array([1.0, 1.0, 1.0])
target_position = np.array([1000.0, 1000.0, 1000.0])
time_steps = np.linspace(0, 100, 1000)
G = 6.67430e-11

def apply_gravity(position, velocity, planetary_data, time_step):
    for _, planet in planetary_data.iterrows():
        planet_mass = planet.get('mass', 0) * 1.989e30
        planet_position = np.array([planet.get('x', 0), planet.get('y', 0), planet.get('z', 0)])
        distance_vector = planet_position - position
        distance = np.linalg.norm(distance_vector)
        if distance > 0:
            gravitational_force = G * spacecraft_mass * planet_mass / distance**2
            gravitational_acceleration = (gravitational_force / spacecraft_mass) * (distance_vector / distance)
            velocity += gravitational_acceleration * time_step
    return velocity

def simulate_trajectory(time_steps, initial_position, velocity, planetary_data):
    positions = []
    position = initial_position.copy()
    for t in time_steps:
        velocity = apply_gravity(position, velocity, planetary_data, time_step=0.1)
        position += velocity * 0.1
        positions.append(position.copy())
    return np.array(positions)

def apply_thrust(position, velocity, thrust_direction, thrust_magnitude, time_step):
    force = thrust_direction * thrust_magnitude
    acceleration = force / spacecraft_mass
    velocity += acceleration * time_step
    position += velocity * time_step
    return position, velocity

def objective(thrust_direction, planetary_data):
    thrust_direction = np.array(thrust_direction, dtype=float)
    position = np.array(initial_position, dtype=float)
    velocity = np.array(initial_velocity, dtype=float)
    fuel_used = 0

    for _ in range(len(time_steps)):
        velocity = apply_gravity(position, velocity, planetary_data, time_step=0.1)
        position, velocity = apply_thrust(position, velocity, thrust_direction, thrust_magnitude, time_step=0.1)
        fuel_used += thrust_magnitude * 0.1

    distance_to_target = np.linalg.norm(position - target_position)
    return distance_to_target + fuel_used

def visualize_trajectory(trajectory, planetary_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Spacecraft Path', color='blue')
    for _, planet in planetary_data.iterrows():
        ax.scatter(planet.get('x', 0), planet.get('y', 0), planet.get('z', 0), label=planet.get('name', 'Unknown'), color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spacecraft Trajectory with Planetary Positions')
    ax.legend()
    plt.show()

def get_asteroid_data():
    url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
    params = {'api_key': 'DEMO_KEY', 'size': 5}
    response = requests.get(url, params=params)
    data = response.json()
    return data['near_earth_objects']

def main():
    planetary_data = load_planetary_data()
    trajectory = simulate_trajectory(time_steps, initial_position, initial_velocity, planetary_data)
    initial_guess = [0.1, 0.1, 0.1]
    result = minimize(objective, initial_guess, args=(planetary_data,), bounds=[(-1, 1), (-1, 1), (-1, 1)])
    optimized_thrust_direction = result.x
    print(f"Optimized Thrust Direction: {optimized_thrust_direction}")
    corrected_positions = []
    position = np.array(initial_position, dtype=float)
    velocity = np.array(initial_velocity, dtype=float)
    for t in time_steps:
        velocity = apply_gravity(position, velocity, planetary_data, time_step=0.1)
        position, velocity = apply_thrust(position, velocity, optimized_thrust_direction, thrust_magnitude, time_step=0.1)
        corrected_positions.append(position.copy())
    corrected_positions = np.array(corrected_positions)
    visualize_trajectory(corrected_positions, planetary_data)
    asteroids = get_asteroid_data()
    print("Detected Asteroids:")
    for asteroid in asteroids:
        print(f"Name: {asteroid['name']}, Diameter: {asteroid['estimated_diameter']['meters']['max']} meters")

if __name__ == "__main__":
    main()
