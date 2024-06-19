# Standard library imports
from typing import NamedTuple

# Third-party imports

# Local library imports
from calibrationpinn.typeAliases import NPArray


class SimulationResults1D(NamedTuple):
    coordinates: NPArray
    predictions: NPArray
    solutions: NPArray
    residuals: NPArray
    relative_errors: NPArray


class SimulationResults2D(NamedTuple):
    coordinates_grid_x: NPArray
    coordinates_grid_y: NPArray
    prediction_grid: NPArray
    solution_grid: NPArray
    residual_grid: NPArray
    relative_error_grid: NPArray
