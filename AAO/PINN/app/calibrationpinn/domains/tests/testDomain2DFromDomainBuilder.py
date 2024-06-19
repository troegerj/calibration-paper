# Standard library imports
import unittest

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains import DomainBuilder2D
from calibrationpinn.domains.tests.testDoubles import (
    FakeDataSelectionFunc,
    FakeInputDataReader2D,
)
from calibrationpinn.inputoutput import PathAdministrator, CSVDataReader
from calibrationpinn.settings import Settings


class TestDomain2DFromBuilderWithSolution(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._num_data_points = 5
        self._num_collocation_points = 4
        self._num_simulation_points_per_edge = 3
        settings = Settings()
        path_admin = PathAdministrator(settings)
        data_reader = CSVDataReader(path_admin)
        self._fake_input_data_reader = FakeInputDataReader2D(data_reader)
        self._fake_observation_data = (
            self._fake_input_data_reader.read_observation_data()
        )
        self._fake_force_data = self._fake_input_data_reader.read_force_data()
        self._fake_solution_data = self._fake_input_data_reader.read_solution_data()
        fake_data_section_func = FakeDataSelectionFunc()
        PRNG_key = jax.random.PRNGKey(0)
        domain_builder = DomainBuilder2D()
        self._sut = domain_builder.build_domain_with_solution(
            input_reader=self._fake_input_data_reader,
            data_selection_func=fake_data_section_func,
            PRNG_key=PRNG_key,
        )

    # training data
    def test_training_data_x_data(self) -> None:
        """
        Test that the data inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_data

        expected = self._fake_observation_data.coordinates
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_data_true_ux(self) -> None:
        """
        Test that the data outputs (displacements in x) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_data_true_ux

        expected = self._fake_observation_data.displacements[:, 0].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_data_true_uy(self) -> None:
        """
        Test that the data outputs (displacements in y) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_data_true_uy

        expected = self._fake_observation_data.displacements[:, 1].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_x_pde(self) -> None:
        """
        Test that the pde inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_pde

        expected = self._fake_observation_data.coordinates
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_pde(self) -> None:
        """
        Test that the pde outputs (zeros) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_pde_true

        expected = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_volume_force(self) -> None:
        """
        Test that the volume force data of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.volume_force

        expected = self._fake_force_data.volume_force
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_coordinates_traction_bc(self) -> None:
        """
        Test that the traction boundary condition inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_traction_bc

        expected = self._fake_force_data.coordinates_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_normals_traction_bc(self) -> None:
        """
        Test that the traction boundary condition normal vectors of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.n_traction_bc

        expected = self._fake_force_data.normals_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_traction_bc(self) -> None:
        """
        Test that the traction boundary condition data of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_traction_bc_true

        expected = self._fake_force_data.traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # validation data
    def test_validation_data_x_data(self) -> None:
        """
        Test that the data inputs (coordinates) of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.x_data

        expected = self._fake_solution_data.coordinates
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_data_true_ux(self) -> None:
        """
        Test that the data outputs (displacements in x) of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_data_true_ux

        expected = self._fake_solution_data.displacements[:, 0].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_data_true_uy(self) -> None:
        """
        Test that the data outputs (displacements in y) of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_data_true_uy

        expected = self._fake_solution_data.displacements[:, 1].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_youngs_moduli_true(self) -> None:
        """
        Test that the Young's modulus data of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_youngs_moduli_true

        expected = self._fake_solution_data.youngs_moduli
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_poissons_ratios_true(self) -> None:
        """
        Test that the Poisson's ratio data of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_poissons_ratios_true

        expected = self._fake_solution_data.poissons_ratios
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # simulation data
    def test_simulation_data_coordinates_grid_x(self) -> None:
        """
        Test that the data input grid (coordinates in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.coordinates_grid_x

        expected = jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_coordinates_grid_y(self) -> None:
        """
        Test that the data input grid (coordinates in y) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.coordinates_grid_y

        expected = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_displacements_grid_x(self) -> None:
        """
        Test that the data output grid (displacements in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.displacements_grid_x

        expected = jnp.array(
            [[31.0, 21.0, 11.0], [31.0, 21.0, 11.0], [31.0, 21.0, 11.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_displacements_grid_y(self) -> None:
        """
        Test that the data output grid (displacements in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.displacements_grid_y

        expected = jnp.array(
            [[101.0, 101.0, 101.0], [201.0, 201.0, 201.0], [301.0, 301.0, 301.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_youngs_moduli_grid(self) -> None:
        """
        Test that the youngs modulus grid of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.youngs_moduli_grid

        expected = jnp.array(
            [[41.0, 31.0, 21.0], [41.0, 31.0, 21.0], [41.0, 31.0, 21.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_poissons_ratios_grid(self) -> None:
        """
        Test that the poissons ratio grid of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.poissons_ratios_grid

        expected = jnp.array(
            [[201.0, 201.0, 201.0], [301.0, 301.0, 301.0], [401.0, 401.0, 401.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestDomain2DFromBuilderWithoutSolution(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._num_data_points = 5
        self._num_collocation_points = 4
        self._num_simulation_points_per_edge = 3
        settings = Settings()
        path_admin = PathAdministrator(settings)
        data_reader = CSVDataReader(path_admin)
        self._fake_input_data_reader = FakeInputDataReader2D(data_reader)
        self._fake_observation_data = (
            self._fake_input_data_reader.read_observation_data()
        )
        self._fake_force_data = self._fake_input_data_reader.read_force_data()
        proportion_training_data = 0.5
        fake_data_section_func = FakeDataSelectionFunc()
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_keys = jax.random.split(PRNG_key, 3)
        self._PRNG_key_data_splitting = PRNG_keys[2]
        domain_builder = DomainBuilder2D()
        self._sut = domain_builder.build_domain_without_solution(
            input_reader=self._fake_input_data_reader,
            proportion_training_data=proportion_training_data,
            data_selection_func=fake_data_section_func,
            PRNG_key=PRNG_key,
        )

    # training data
    def test_training_data_x_data(self) -> None:
        """
        Test that the data inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_data

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting, self._fake_observation_data.coordinates
        )[:2]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_data_true_ux(self) -> None:
        """
        Test that the data outputs (displacements in x) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_data_true_ux

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting,
            self._fake_observation_data.displacements[:, 0],
        )[:2].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_data_true_uy(self) -> None:
        """
        Test that the data outputs (displacements in y) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_data_true_uy

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting,
            self._fake_observation_data.displacements[:, 1],
        )[:2].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_x_pde(self) -> None:
        """
        Test that the pde inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_pde

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting, self._fake_observation_data.coordinates
        )[:2]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_y_pde(self) -> None:
        """
        Test that the pde outputs (zeros) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_pde_true

        expected = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_volume_force(self) -> None:
        """
        Test that the volume force data of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.volume_force

        expected = self._fake_force_data.volume_force
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_coordinates_traction_bc(self) -> None:
        """
        Test that the traction boundary condition inputs (coordinates) of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.x_traction_bc

        expected = self._fake_force_data.coordinates_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_normals_traction_bc(self) -> None:
        """
        Test that the traction boundary condition normal vectors of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.n_traction_bc

        expected = self._fake_force_data.normals_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_training_data_traction_bc(self) -> None:
        """
        Test that the traction boundary conditions data of the training data set are generated correctly.
        """
        train_data = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = train_data.y_traction_bc_true

        expected = self._fake_force_data.traction_bc
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

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting, self._fake_observation_data.coordinates
        )[2:]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_data_true_ux(self) -> None:
        """
        Test that the data outputs (displacements in x) of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_data_true_ux

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting,
            self._fake_observation_data.displacements[:, 0],
        )[2:].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_data_true_uy(self) -> None:
        """
        Test that the data outputs (displacements in y) of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_data_true_uy

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting,
            self._fake_observation_data.displacements[:, 1],
        )[2:].reshape((-1, 1))
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_youngs_moduli_true(self) -> None:
        """
        Test that the Young's modulus data of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_youngs_moduli_true

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting, self._fake_observation_data.youngs_moduli
        )[2:]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_validation_data_y_poissons_ratios_true(self) -> None:
        """
        Test that the poissons ratio data of the validation data set are generated correctly.
        """
        valid_data = self._sut.generate_validation_data(
            num_data_points=self._num_data_points
        )
        actual = valid_data.y_poissons_ratios_true

        expected = jax.random.permutation(
            self._PRNG_key_data_splitting, self._fake_observation_data.poissons_ratios
        )[2:]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # simulation data
    def test_simulation_data_coordinates_grid_x(self) -> None:
        """
        Test that the data input grid (coordinates in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.coordinates_grid_x

        expected = jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_coordinates_grid_y(self) -> None:
        """
        Test that the data input grid (coordinates in y) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.coordinates_grid_y

        expected = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_displacements_grid_x(self) -> None:
        """
        Test that the data output grid (displacements in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.displacements_grid_x

        expected = jnp.array(
            [[30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [30.0, 20.0, 10.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_displacements_grid_y(self) -> None:
        """
        Test that the data output grid (displacements in x) of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.displacements_grid_y

        expected = jnp.array(
            [[100.0, 100.0, 100.0], [200.0, 200.0, 200.0], [300.0, 300.0, 300.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_youngs_moduli_grid(self) -> None:
        """
        Test that the youngs modulus grid of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.youngs_moduli_grid

        expected = jnp.array(
            [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [40.0, 30.0, 20.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_simulation_data_poissons_ratios_grid(self) -> None:
        """
        Test that the poissons ratio grid of the simulation data set is generated correctly.
        """
        sim_data = self._sut.generate_simulation_data(
            num_data_points_per_edge=self._num_simulation_points_per_edge
        )
        actual = sim_data.poissons_ratios_grid

        expected = jnp.array(
            [[200.0, 200.0, 200.0], [300.0, 300.0, 300.0], [400.0, 400.0, 400.0]]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)
