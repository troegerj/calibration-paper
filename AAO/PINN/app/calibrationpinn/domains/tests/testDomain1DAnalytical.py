# Standard library imports
import unittest

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains import Domain1DAnalytical


class TestDomain1DAnalytical(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._lenght = 12.0
        self._traction = 2.0
        self._volume_force = 4.0
        self._youngs_modulus = 1.0
        self._num_data_points = 3
        self._num_collocation_points = 4
        self._sut = Domain1DAnalytical(
            length=self._lenght,
            traction=self._traction,
            volume_force=self._volume_force,
            youngs_modulus=self._youngs_modulus,
        )

    # training data
    def test_training_data_x_data(self) -> None:
        """
        Test that the data inputs for the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_data

        expected = jnp.array([[0.0], [6.0], [12.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_data_true(self) -> None:
        """
        Test that the data outputs (displacements) for the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_data_true

        expected = jnp.array([[0.0], [228.0], [312.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_x_pde(self) -> None:
        """
        Test that the pde inputs for the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_pde

        expected = jnp.array([[0.0], [4.0], [8.0], [12.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_pde(self) -> None:
        """
        Test that the pde outputs for the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_pde_true

        expected = jnp.array([[0.0], [0.0], [0.0], [0.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_volume_force(self) -> None:
        """
        Test that the volume force data for the training data set is generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.volume_force

        expected = jnp.array([self._volume_force])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_coordinates_traction_bc(self) -> None:
        """
        Test that the coordinates of the traction boundary condition data for the training data set is generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_traction_bc

        expected = jnp.array([[self._lenght]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data__traction_bc(self) -> None:
        """
        Test that the traction boundary condition data for the training data set is generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_traction_bc_true

        expected = jnp.array([[self._traction]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # validation data
    def test_validation_data_x_data(self) -> None:
        """
        Test that the data inputs for the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.x_data

        expected = jnp.array([[0.0], [6.0], [12.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_data_true(self) -> None:
        """
        Test that the data outputs (displacements) for the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_data_true

        expected = jnp.array([[0.0], [228.0], [312.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_youngs_moduli_true(self) -> None:
        """
        Test that the youngs modulus outputs for the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_youngs_moduli_true

        expected = jnp.array(
            [[self._youngs_modulus], [self._youngs_modulus], [self._youngs_modulus]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # simulation data
    def test_simulation_data_coordinates(self) -> None:
        """
        Test that the data inputs (coordinates) for the simulation data set are generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points=self._num_data_points
        )
        actual = sim_data.coordinates

        expected = jnp.array([[0.0], [6.0], [12.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_displacements(self) -> None:
        """
        Test that the data outputs (displacements) for the simulation data set are generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points=self._num_data_points
        )
        actual = sim_data.displacements

        expected = jnp.array([[0.0], [228.0], [312.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_youngs_moduli(self) -> None:
        """
        Test that the youngs moduli for the simulation data set are generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points=self._num_data_points
        )
        actual = sim_data.youngs_moduli

        expected = jnp.array(
            [[self._youngs_modulus], [self._youngs_modulus], [self._youngs_modulus]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)
