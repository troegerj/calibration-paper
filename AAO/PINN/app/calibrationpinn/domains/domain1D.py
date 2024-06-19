# Standard library imports
from typing import NamedTuple

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.domains.dataselection import DataSelectionFuncProtocol
from calibrationpinn.domains.inputreader import (
    ForceData1D,
    ObservationData1D,
    SolutionData1D,
)
from calibrationpinn.domains.splitDataFunc import DataSplittingFuncProtocol
from calibrationpinn.typeAliases import PRNGKey
from calibrationpinn.utilities.typeConversionFuncs import numpy_to_jax_numpy


class TrainingData1D(NamedTuple):
    x_data: jnp.ndarray
    y_data_true: jnp.ndarray
    x_pde: jnp.ndarray
    y_pde_true: jnp.ndarray
    volume_force: jnp.ndarray
    x_traction_bc: jnp.ndarray
    y_traction_bc_true: jnp.ndarray


class ValidationData1D(NamedTuple):
    x_data: jnp.ndarray
    y_data_true: jnp.ndarray
    y_youngs_moduli_true: jnp.ndarray


class SimulationData1D(NamedTuple):
    coordinates: jnp.ndarray
    displacements: jnp.ndarray
    youngs_moduli: jnp.ndarray


class Domain1DWithSolution:
    def __init__(
        self,
        observation_data: ObservationData1D,
        force_data: ForceData1D,
        solution_data: SolutionData1D,
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
    ) -> TrainingData1D:
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
        coordinates_pde = self._data_selection_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            num_collocation_points,
            self._PRNG_key_train_data,
        )
        volume_force = numpy_to_jax_numpy(self._force_data.volume_force)
        coordinates_traction_bc = numpy_to_jax_numpy(
            self._force_data.coordinates_traction_bc
        )
        traction_bc = numpy_to_jax_numpy(self._force_data.traction_bc)
        return TrainingData1D(
            x_data=coordinates_data,
            y_data_true=displacements,
            x_pde=coordinates_pde,
            y_pde_true=jnp.zeros((num_collocation_points, 1)),
            volume_force=volume_force,
            x_traction_bc=coordinates_traction_bc,
            y_traction_bc_true=traction_bc,
        )

    def generate_validation_data(self, num_data_points: int) -> ValidationData1D:
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
        youngs_moduli = self._data_selection_func(
            numpy_to_jax_numpy(self._solution_data.youngs_moduli),
            num_data_points,
            self._PRNG_key_valid_data,
        )
        return ValidationData1D(
            x_data=coordinates,
            y_data_true=displacements,
            y_youngs_moduli_true=youngs_moduli,
        )

    def generate_simulation_data(self) -> SimulationData1D:
        return SimulationData1D(
            coordinates=numpy_to_jax_numpy(self._solution_data.coordinates),
            displacements=numpy_to_jax_numpy(self._solution_data.displacements),
            youngs_moduli=numpy_to_jax_numpy(self._solution_data.youngs_moduli),
        )


class Domain1DWithoutSolution:
    def __init__(
        self,
        observation_data: ObservationData1D,
        force_data: ForceData1D,
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
    ) -> TrainingData1D:
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
        coordinates_pde = self._data_selection_func(
            proportion_coordinates,
            num_collocation_points,
            self._PRNG_key_train_data,
        )
        volume_force = self._force_data.volume_force
        coordinates_traction_bc = numpy_to_jax_numpy(
            self._force_data.coordinates_traction_bc
        )
        traction_bc = numpy_to_jax_numpy(self._force_data.traction_bc)
        return TrainingData1D(
            x_data=coordinates_data,
            y_data_true=displacements,
            x_pde=coordinates_pde,
            y_pde_true=jnp.zeros((num_collocation_points, 1)),
            volume_force=jnp.array([volume_force]),
            x_traction_bc=coordinates_traction_bc,
            y_traction_bc_true=traction_bc,
        )

    def generate_validation_data(self, num_data_points: int) -> ValidationData1D:
        _, proportion_coordinates = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.coordinates),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        (
            _,
            proportion_displacements,
        ) = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.displacements),
            self._proportion_train_data,
            self._PRNG_key_data_splitting,
        )
        (
            _,
            proportion_youngs_moduli,
        ) = self._split_data_func(
            numpy_to_jax_numpy(self._observation_data.youngs_moduli),
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
        youngs_moduli = self._data_selection_func(
            proportion_youngs_moduli,
            num_data_points,
            self._PRNG_key_valid_data,
        )
        return ValidationData1D(
            x_data=coordinates,
            y_data_true=displacements,
            y_youngs_moduli_true=youngs_moduli,
        )

    def generate_simulation_data(self) -> SimulationData1D:
        return SimulationData1D(
            coordinates=numpy_to_jax_numpy(self._observation_data.coordinates),
            displacements=numpy_to_jax_numpy(self._observation_data.displacements),
            youngs_moduli=numpy_to_jax_numpy(self._observation_data.youngs_moduli),
        )
