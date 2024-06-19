# Standard library imports
from typing import TypeAlias, NamedTuple

# Third-party imports
from jax import vmap
import jax
import jax.numpy as jnp
import numpy as np

# Local library imports
from calibrationpinn.domains import SimulationData1D
from calibrationpinn.simulations.plotters import Domain1DPlotter, Domain1DPlotterConfig
from calibrationpinn.training.pinns.momentumEquation1D import (
    MomentumEquation1DParameters,
    MomentumEquation1DModels,
)
from calibrationpinn.simulations.simulationResults import SimulationResults1D
from calibrationpinn.typeAliases import HKTransformed, JNPArray, JNPPyTree
from calibrationpinn.utilities.typeConversionFuncs import jax_numpy_to_numpy

Parameters: TypeAlias = MomentumEquation1DParameters
Models: TypeAlias = MomentumEquation1DModels
SimulationData: TypeAlias = SimulationData1D
PlotterConfig: TypeAlias = Domain1DPlotterConfig


class Simulator1D:
    def __init__(self, plotter: Domain1DPlotter) -> None:
        self._plotter = plotter

    def simulate_displacements(
        self,
        parameters: Parameters,
        models: Models,
        simulation_data: SimulationData,
        plotter_config: PlotterConfig,
    ) -> None:
        simulation_results = self._simulate_displacements(
            parameters, models, simulation_data
        )
        self._plot_results(simulation_results, plotter_config)

    def _simulate_displacements(
        self,
        parameters: Parameters,
        models: Models,
        simulation_data: SimulationData,
    ) -> SimulationResults1D:
        solutions = simulation_data.displacements
        predictions = self._calculate_predictions(
            parameters.net_u, models.net_u, simulation_data
        )
        residuals = self._calculate_residuals(predictions, solutions)
        relative_errors = self._calculate_relative_errors(predictions, solutions)
        return SimulationResults1D(
            coordinates=jax_numpy_to_numpy(simulation_data.coordinates),
            predictions=jax_numpy_to_numpy(predictions),
            solutions=jax_numpy_to_numpy(solutions),
            residuals=jax_numpy_to_numpy(residuals),
            relative_errors=jax_numpy_to_numpy(relative_errors),
        )

    def simulate_parameter(
        self,
        parameters: Parameters,
        models: Models,
        parameter_estimate: float,
        simulation_data: SimulationData,
        plotter_config: PlotterConfig,
    ) -> None:
        simulation_results = self._simulate_parameter(
            parameters, models, parameter_estimate, simulation_data
        )
        self._plot_results(simulation_results, plotter_config)

    def _simulate_parameter(
        self,
        parameters: Parameters,
        models: Models,
        parameter_estimate: float,
        simulation_data: SimulationData,
    ) -> SimulationResults1D:
        solutions = simulation_data.youngs_moduli
        predictions = (
            1.0
            + self._calculate_predictions(
                parameters.youngs_modulus_correction,
                models.youngs_modulus_correction,
                simulation_data,
            )
        ) * parameter_estimate
        residuals = self._calculate_residuals(predictions, solutions)
        relative_errors = self._calculate_relative_errors(predictions, solutions)
        return SimulationResults1D(
            coordinates=jax_numpy_to_numpy(simulation_data.coordinates),
            predictions=jax_numpy_to_numpy(predictions),
            solutions=jax_numpy_to_numpy(solutions),
            residuals=jax_numpy_to_numpy(residuals),
            relative_errors=jax_numpy_to_numpy(relative_errors),
        )

    def _calculate_predictions(
        self,
        params_net: JNPPyTree,
        net: HKTransformed,
        simulation_data: SimulationData,
    ) -> JNPArray:
        coordinates = simulation_data.coordinates
        predictions = self._apply_net_to_coordinates(params_net, net, coordinates)
        return predictions

    def _apply_net_to_coordinates(
        self, params_net: JNPPyTree, net: HKTransformed, coordinates: JNPArray
    ) -> JNPArray:
        vmap_net = lambda single_input: net.apply(params_net, single_input)
        predictions = vmap(vmap_net)(coordinates)
        return predictions

    def _calculate_residuals(
        self, predictions: JNPArray, solutions: JNPArray
    ) -> JNPArray:
        return predictions - solutions

    def _calculate_relative_errors(
        self, predictions: JNPArray, solutions: JNPArray
    ) -> JNPArray:
        residuals = self._calculate_residuals(predictions, solutions)
        return residuals / solutions

    def _plot_results(
        self, simulation_results: SimulationResults1D, plotter_config: PlotterConfig
    ) -> None:
        self._plotter.plot(simulation_results, plotter_config)
