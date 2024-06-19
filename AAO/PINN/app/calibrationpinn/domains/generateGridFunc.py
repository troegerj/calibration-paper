# Standard library imports

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.typeAliases import NPArray


def generate_grid(coordinates: NPArray, num_points_per_edge: int) -> list[NPArray]:
    coordinates_x = coordinates[:, 0]
    coordinates_y = coordinates[:, 1]
    grid_coordinates_x = np.linspace(
        np.amin(coordinates_x), np.amax(coordinates_x), num=num_points_per_edge
    )
    grid_coordinates_y = np.linspace(
        np.amin(coordinates_y), np.amax(coordinates_y), num=num_points_per_edge
    )
    return np.meshgrid(grid_coordinates_x, grid_coordinates_y)
