# Standard library imports
from typing import NamedTuple

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.domains.domain1D import (
    TrainingData1D,
    ValidationData1D,
    SimulationData1D,
)
from calibrationpinn.typeAliases import JNPArray


class Domain1DAnalytical:
    def __init__(
        self,
        length,
        traction,
        volume_force,
        youngs_modulus,
    ) -> None:
        self._length = length
        self._traction = traction
        self._volume_force = volume_force
        self._youngs_modulus = youngs_modulus

    def generate_training_data(
        self, num_data_points: int, num_collocation_points: int
    ) -> TrainingData1D:
        x_data = self._arrange_points(num_data_points)
        y_data_true = self._displacement_func(x_data)
        x_pde = self._arrange_points(num_collocation_points)
        y_pde_true = self._generate_zeros(num_collocation_points)
        return TrainingData1D(
            x_data=x_data,
            y_data_true=y_data_true,
            x_pde=x_pde,
            y_pde_true=y_pde_true,
            volume_force=jnp.array([self._volume_force]),
            x_traction_bc=jnp.array([[self._length]]),
            y_traction_bc_true=jnp.array([[self._traction]]),
        )

    def generate_validation_data(self, num_data_points: int) -> ValidationData1D:
        x_data = self._arrange_points(num_data_points)
        y_data_true = self._displacement_func(x_data)
        youngs_moduli = self._generate_youngs_moduli(num_data_points)
        return ValidationData1D(
            x_data=x_data,
            y_data_true=y_data_true,
            y_youngs_moduli_true=youngs_moduli,
        )

    def generate_simulation_data(self, num_data_points: int) -> SimulationData1D:
        coordinates = self._arrange_points(num_data_points)
        displacements = self._displacement_func(coordinates)
        youngs_moduli = self._generate_youngs_moduli(num_data_points)
        return SimulationData1D(
            coordinates=coordinates,
            displacements=displacements,
            youngs_moduli=youngs_moduli,
        )

    def _arrange_points(self, num_points: int, endpoint: bool = True) -> JNPArray:
        return jnp.linspace(
            0.0, self._length, num=num_points, endpoint=endpoint
        ).reshape((-1, 1))

    def _generate_zeros(self, num_points: int) -> JNPArray:
        return jnp.zeros((num_points, 1))

    def _displacement_func(self, coordinates: JNPArray) -> JNPArray:
        x = coordinates
        return (self._traction / self._youngs_modulus) * x + (
            self._volume_force / self._youngs_modulus
        ) * (self._length * x - 1 / 2 * x**2)

    def _generate_youngs_moduli(self, num_points: int) -> JNPArray:
        return jnp.full((num_points, 1), self._youngs_modulus)
