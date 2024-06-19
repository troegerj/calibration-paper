# Standard library imports
from typing import TypeAlias, Union

# Third-party imports
from jax import vmap
import jax.numpy as jnp

# Local library imports
from calibrationpinn.domains import SimulationData2D
from calibrationpinn.simulations.plotters import Domain2DPlotter, Domain2DPlotterConfig
from calibrationpinn.training.pinns.momentumEquation2D_E_nu import (
    MomentumEquation2DParameters_E_nu,
    MomentumEquation2DModels_E_nu,
)
from calibrationpinn.training.pinns.momentumEquation2D_K_G import (
    MomentumEquation2DParameters_K_G,
    MomentumEquation2DModels_K_G,
)
from calibrationpinn.simulations.simulationResults import SimulationResults2D
from calibrationpinn.typeAliases import HKTransformed, JNPArray, JNPPyTree
from calibrationpinn.utilities.typeConversionFuncs import jax_numpy_to_numpy

Parameters: TypeAlias = Union[
    MomentumEquation2DParameters_E_nu, MomentumEquation2DParameters_K_G
]
Models: TypeAlias = Union[MomentumEquation2DModels_E_nu, MomentumEquation2DModels_K_G]
SimulationData: TypeAlias = SimulationData2D
PlotterConfig: TypeAlias = Domain2DPlotterConfig


class Simulator2D:
    def __init__(self, plotter: Domain2DPlotter) -> None:
        self._plotter = plotter

    def simulate_displacements(
        self,
        parameters: Parameters,
        models: Models,
        simulation_data: SimulationData,
        plotter_config_displacements_x: PlotterConfig,
        plotter_config_displacements_y: PlotterConfig,
    ) -> None:
        solution_grid_displacements_x = simulation_data.displacements_grid_x
        simulation_results_displacements_x = self._simulate_displacement_grid(
            parameters.net_ux,
            models.net_ux,
            solution_grid_displacements_x,
            simulation_data,
        )
        solution_grid_displacements_y = simulation_data.displacements_grid_y
        simulation_results_displacements_y = self._simulate_displacement_grid(
            parameters.net_uy,
            models.net_uy,
            solution_grid_displacements_y,
            simulation_data,
        )
        self._plot_results(
            simulation_results_displacements_x, plotter_config_displacements_x
        )
        self._plot_results(
            simulation_results_displacements_y, plotter_config_displacements_y
        )

    def _simulate_displacement_grid(
        self,
        parameters_net: JNPPyTree,
        net: HKTransformed,
        solution_grid: JNPArray,
        simulation_data: SimulationData,
    ) -> SimulationResults2D:
        prediction_grid = self._calculate_prediction_grid(
            parameters_net, net, simulation_data
        )
        residual_grid = self._calculate_residual_grid(prediction_grid, solution_grid)
        relative_error_grid = self._calculate_relative_error_grid(
            prediction_grid, solution_grid
        )
        return SimulationResults2D(
            coordinates_grid_x=jax_numpy_to_numpy(simulation_data.coordinates_grid_x),
            coordinates_grid_y=jax_numpy_to_numpy(simulation_data.coordinates_grid_y),
            prediction_grid=jax_numpy_to_numpy(prediction_grid),
            solution_grid=jax_numpy_to_numpy(solution_grid),
            residual_grid=jax_numpy_to_numpy(residual_grid),
            relative_error_grid=jax_numpy_to_numpy(relative_error_grid),
        )

    def simulate_parameter(
        self,
        parameters_net: JNPPyTree,
        net: HKTransformed,
        parameter_estimate: float,
        solution_grid: JNPArray,
        simulation_data: SimulationData,
        plotter_config: PlotterConfig,
    ) -> None:
        simulation_results = self._simulate_parameter_grid(
            parameters_net, net, parameter_estimate, solution_grid, simulation_data
        )
        self._plot_results(simulation_results, plotter_config)

    def _simulate_parameter_grid(
        self,
        parameters_net: JNPPyTree,
        net: HKTransformed,
        parameter_estimate: float,
        solution_grid: JNPArray,
        simulation_data: SimulationData,
    ) -> SimulationResults2D:
        prediction_grid = (
            1.0 + self._calculate_prediction_grid(parameters_net, net, simulation_data)
        ) * parameter_estimate
        residual_grid = self._calculate_residual_grid(prediction_grid, solution_grid)
        relative_error_grid = self._calculate_relative_error_grid(
            prediction_grid, solution_grid
        )
        return SimulationResults2D(
            coordinates_grid_x=jax_numpy_to_numpy(simulation_data.coordinates_grid_x),
            coordinates_grid_y=jax_numpy_to_numpy(simulation_data.coordinates_grid_y),
            prediction_grid=jax_numpy_to_numpy(prediction_grid),
            solution_grid=jax_numpy_to_numpy(solution_grid),
            residual_grid=jax_numpy_to_numpy(residual_grid),
            relative_error_grid=jax_numpy_to_numpy(relative_error_grid),
        )

    def _calculate_prediction_grid(
        self,
        parameters_net: JNPPyTree,
        net: HKTransformed,
        simulation_data: SimulationData,
    ) -> JNPArray:
        coordinates_grid = self._stack_coordinate_grids(simulation_data)
        prediction_grid = self._apply_net_to_coordinates_grid(
            parameters_net, net, coordinates_grid
        )
        return prediction_grid

    def _stack_coordinate_grids(self, simulation_data: SimulationData) -> JNPArray:
        coordinates_grid_x = simulation_data.coordinates_grid_x
        coordinates_grid_y = simulation_data.coordinates_grid_y
        return jnp.stack((coordinates_grid_x, coordinates_grid_y), axis=2)

    def _apply_net_to_coordinates_grid(
        self, parameters_net: JNPPyTree, net: HKTransformed, coordinates_grid: JNPArray
    ) -> JNPArray:
        batched_coordinates = coordinates_grid.reshape((-1, 1, 2))
        vmap_net = lambda single_input: net.apply(parameters_net, single_input)
        batched_predictions = vmap(vmap_net)(batched_coordinates)
        shape_coordinates_grid = coordinates_grid.shape
        shape_prediction_grid = (shape_coordinates_grid[0], shape_coordinates_grid[1])
        return batched_predictions.reshape((shape_prediction_grid))

    def _calculate_residual_grid(
        self, prediction_grid: JNPArray, solution_grid: JNPArray
    ) -> JNPArray:
        return prediction_grid - solution_grid

    def _calculate_relative_error_grid(
        self, prediction_grid: JNPArray, solution_grid: JNPArray
    ) -> JNPArray:
        residual_grid = self._calculate_residual_grid(prediction_grid, solution_grid)
        return residual_grid / solution_grid

    def _plot_results(
        self, simulation_results: SimulationResults2D, plotter_config: PlotterConfig
    ) -> None:
        self._plotter.plot(simulation_results, plotter_config)
