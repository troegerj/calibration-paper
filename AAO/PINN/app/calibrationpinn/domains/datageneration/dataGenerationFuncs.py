# Standard library imports

# Third-party imports
from scipy.interpolate import griddata

# Local library imports
from calibrationpinn.typeAliases import NPArray


def interpolate_on_grid(
    coordinates_solution: NPArray,
    values_solution: NPArray,
    coordinates_grid_x: NPArray,
    coordinates_grid_y: NPArray,
    method: str = "linear",
) -> NPArray:
    values_solution = values_solution.reshape((-1,))
    return griddata(
        coordinates_solution,
        values_solution,
        (coordinates_grid_x, coordinates_grid_y),
        method,
    )
