# Standard library imports
from typing import NamedTuple

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.domains.datageneration import interpolate_on_grid
from calibrationpinn.domains.dataselection import DataSelectionFuncProtocol
from calibrationpinn.domains.generateGridFunc import generate_grid
from calibrationpinn.domains.inputreader import (
    ForceData2D,
    ObservationData2D,
    SolutionData2D,
)
from calibrationpinn.domains.splitDataFunc import DataSplittingFuncProtocol
from calibrationpinn.typeAliases import PRNGKey
from calibrationpinn.utilities.typeConversionFuncs import numpy_to_jax_numpy


class TrainingData2D(NamedTuple):
    x_data: jnp.ndarray
    y_data_true_ux: jnp.ndarray
    y_data_true_uy: jnp.ndarray
    x_pde: jnp.ndarray
    y_pde_true: jnp.ndarray
    volume_force: jnp.ndarray
    x_traction_bc: jnp.ndarray
    n_traction_bc: jnp.ndarray
    y_traction_bc_true: jnp.ndarray


class ValidationData2D(NamedTuple):
    x_data: jnp.ndarray
    y_data_true_ux: jnp.ndarray
    y_data_true_uy: jnp.ndarray
    y_youngs_moduli_true: jnp.ndarray
    y_poissons_ratios_true: jnp.ndarray


class SimulationData2D(NamedTuple):
    coordinates_grid_x: jnp.ndarray
    coordinates_grid_y: jnp.ndarray
    displacements_grid_x: jnp.ndarray
    displacements_grid_y: jnp.ndarray
    youngs_moduli_grid: jnp.ndarray
    poissons_ratios_grid: jnp.ndarray


class Domain2DWithSolution:
    def __init__(
        self,
        observation_data: ObservationData2D,
        force_data: ForceData2D,
        solution_data: SolutionData2D,
        data_selection_func: DataSelectionFuncProtocol,
        PRNG_key: PRNGKey,
    ) -> None:
        self._observation_data = observation_data
        self._force_data = force_data
        self._solution_data = solution_data
        self._data_selection_func = data_selection_func
        PRNG_keys = jax.random.split(PRNG_key, 2)
        self._PRNG_key_train_data = PRNG_keys[0]
        self._PRNG_key_valid_data = PRNG_keys[1]

    def generate_training_data(
        self, num_data_points: int, num_collocation_points: int
    ) -> TrainingData2D:
        coordinates_data = self._data_selection_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            num_data_points,
            self._PRNG_key_train_data,
        )
        displacements = self._data_selection_func(
            numpy_to_jax_numpy(self._observation_data.displacements),
            num_data_points,
            self._PRNG_key_train_data,
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        coordinates_pde = self._data_selection_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            num_collocation_points,
            self._PRNG_key_train_data,
        )
        volume_force = numpy_to_jax_numpy(self._force_data.volume_force)
        coordinates_traction_bc = numpy_to_jax_numpy(
            self._force_data.coordinates_traction_bc
        )
        nomals_traction_bc = numpy_to_jax_numpy(self._force_data.normals_traction_bc)
        traction_bc = numpy_to_jax_numpy(self._force_data.traction_bc)
        return TrainingData2D(
            x_data=coordinates_data,
            y_data_true_ux=displacements_x,
            y_data_true_uy=displacements_y,
            x_pde=coordinates_pde,
            y_pde_true=jnp.zeros((num_collocation_points, 2)),
            volume_force=volume_force,
            x_traction_bc=coordinates_traction_bc,
            n_traction_bc=nomals_traction_bc,
            y_traction_bc_true=traction_bc,
        )

    def generate_validation_data(self, num_data_points: int) -> ValidationData2D:
        coordinates = self._data_selection_func(
            numpy_to_jax_numpy(self._solution_data.coordinates),
            num_data_points,
            self._PRNG_key_valid_data,
        )
        displacements = self._data_selection_func(
            numpy_to_jax_numpy(self._solution_data.displacements),
            num_data_points,
            self._PRNG_key_valid_data,
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        youngs_moduli = self._data_selection_func(
            numpy_to_jax_numpy(self._solution_data.youngs_moduli),
            num_data_points,
            self._PRNG_key_valid_data,
        )
        poissons_ratios = self._data_selection_func(
            numpy_to_jax_numpy(self._solution_data.poissons_ratios),
            num_data_points,
            self._PRNG_key_valid_data,
        )
        return ValidationData2D(
            x_data=coordinates,
            y_data_true_ux=displacements_x,
            y_data_true_uy=displacements_y,
            y_youngs_moduli_true=youngs_moduli,
            y_poissons_ratios_true=poissons_ratios,
        )

    def generate_simulation_data(
        self, num_data_points_per_edge: int
    ) -> SimulationData2D:
        coordinates = self._solution_data.coordinates
        coordinates_grid_x, coordinates_grid_y = generate_grid(
            coordinates, num_data_points_per_edge
        )
        displacements = self._solution_data.displacements
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        displacements_grid_x = interpolate_on_grid(
            coordinates,
            displacements_x,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        displacements_grid_y = interpolate_on_grid(
            coordinates,
            displacements_y,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        youngs_moduli_grid = interpolate_on_grid(
            coordinates,
            self._solution_data.youngs_moduli,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        poissons_ratios_grid = interpolate_on_grid(
            coordinates,
            self._solution_data.poissons_ratios,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        return SimulationData2D(
            coordinates_grid_x=numpy_to_jax_numpy(coordinates_grid_x),
            coordinates_grid_y=numpy_to_jax_numpy(coordinates_grid_y),
            displacements_grid_x=numpy_to_jax_numpy(displacements_grid_x),
            displacements_grid_y=numpy_to_jax_numpy(displacements_grid_y),
            youngs_moduli_grid=numpy_to_jax_numpy(youngs_moduli_grid),
            poissons_ratios_grid=numpy_to_jax_numpy(poissons_ratios_grid),
        )


class Domain2DWithoutSolution:
    def __init__(
        self,
        observation_data: ObservationData2D,
        force_data: ForceData2D,
        proportion_train_data: float,
        split_data_func: DataSplittingFuncProtocol,
        data_selection_func: DataSelectionFuncProtocol,
        PRNG_key: PRNGKey,
    ) -> None:
        self._observation_data = observation_data
        self._force_data = force_data
        self._proportion_train_data = proportion_train_data
        self._split_data_func = split_data_func
        self._data_selection_func = data_selection_func
        PRNG_keys = jax.random.split(PRNG_key, 3)
        self._PRNG_key_train_data = PRNG_keys[0]
        self._PRNG_key_valid_data = PRNG_keys[1]
        self._PRNG_key_data_splitting = PRNG_keys[2]

    def generate_training_data(
        self, num_data_points: int, num_collocation_points: int
    ) -> TrainingData2D:
        proportion_coordinates, _ = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        proportion_displacements, _ = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.displacements),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        coordinates_data = self._data_selection_func(
            proportion_coordinates,
            num_data_points,
            self._PRNG_key_train_data,
        )
        displacements = self._data_selection_func(
            proportion_displacements,
            num_data_points,
            self._PRNG_key_train_data,
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        coordinates_pde = self._data_selection_func(
            proportion_coordinates,
            num_collocation_points,
            self._PRNG_key_train_data,
        )
        volume_force = numpy_to_jax_numpy(self._force_data.volume_force)
        coordinates_traction_bc = numpy_to_jax_numpy(
            self._force_data.coordinates_traction_bc
        )
        nomals_traction_bc = numpy_to_jax_numpy(self._force_data.normals_traction_bc)
        traction_bc = numpy_to_jax_numpy(self._force_data.traction_bc)
        return TrainingData2D(
            x_data=coordinates_data,
            y_data_true_ux=displacements_x,
            y_data_true_uy=displacements_y,
            x_pde=coordinates_pde,
            y_pde_true=jnp.zeros((num_collocation_points, 2)),
            volume_force=volume_force,
            x_traction_bc=coordinates_traction_bc,
            n_traction_bc=nomals_traction_bc,
            y_traction_bc_true=traction_bc,
        )

    def generate_validation_data(self, num_data_points: int) -> ValidationData2D:
        _, proportion_coordinates = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        _, proportion_displacements = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.displacements),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        _, proportion_youngs_moduli = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.youngs_moduli),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        _, proportion_poissons_ratios = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.poissons_ratios),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        coordinates = self._data_selection_func(
            proportion_coordinates,
            num_data_points,
            self._PRNG_key_valid_data,
        )
        displacements = self._data_selection_func(
            proportion_displacements,
            num_data_points,
            self._PRNG_key_valid_data,
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        youngs_moduli = self._data_selection_func(
            proportion_youngs_moduli,
            num_data_points,
            self._PRNG_key_valid_data,
        )
        poissons_ratios = self._data_selection_func(
            proportion_poissons_ratios,
            num_data_points,
            self._PRNG_key_valid_data,
        )
        return ValidationData2D(
            x_data=coordinates,
            y_data_true_ux=displacements_x,
            y_data_true_uy=displacements_y,
            y_youngs_moduli_true=youngs_moduli,
            y_poissons_ratios_true=poissons_ratios,
        )

    def generate_simulation_data(
        self, num_data_points_per_edge: int
    ) -> SimulationData2D:
        coordinates = self._observation_data.coordinates
        coordinates_grid_x, coordinates_grid_y = generate_grid(
            coordinates, num_data_points_per_edge
        )
        displacements = self._observation_data.displacements
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        displacements_grid_x = interpolate_on_grid(
            coordinates,
            displacements_x,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        displacements_grid_y = interpolate_on_grid(
            coordinates,
            displacements_y,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        youngs_moduli_grid = interpolate_on_grid(
            coordinates,
            self._observation_data.youngs_moduli,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        poissons_ratios_grid = interpolate_on_grid(
            coordinates,
            self._observation_data.poissons_ratios,
            coordinates_grid_x,
            coordinates_grid_y,
        )
        return SimulationData2D(
            coordinates_grid_x=numpy_to_jax_numpy(coordinates_grid_x),
            coordinates_grid_y=numpy_to_jax_numpy(coordinates_grid_y),
            displacements_grid_x=numpy_to_jax_numpy(displacements_grid_x),
            displacements_grid_y=numpy_to_jax_numpy(displacements_grid_y),
            youngs_moduli_grid=numpy_to_jax_numpy(youngs_moduli_grid),
            poissons_ratios_grid=numpy_to_jax_numpy(poissons_ratios_grid),
        )
