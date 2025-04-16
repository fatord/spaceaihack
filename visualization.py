import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectory(trajectory, planetary_data, save_path=None):
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
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
