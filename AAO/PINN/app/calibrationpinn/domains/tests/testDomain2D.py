# Standard library imports
import unittest

# Third-party imports
import jax

# Local library imports
from calibrationpinn.assertions import (
    assert_equal,
    assert_equal_PRNGKeys,
)
from calibrationpinn.domains.domain2D import (
    Domain2DWithSolution,
    Domain2DWithoutSolution,
)
from calibrationpinn.domains.tests.testDoubles import (
    FakeForceData2D,
    FakeObservationData2D,
    FakeSolutionData2D,
    SpyDataSelectionFunc,
    SpyDataSplittingFunc,
)
from calibrationpinn.inputoutput import PathAdministrator, CSVDataReader
from calibrationpinn.settings import Settings


class TestDomain2DWithSolution(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        path_admin = PathAdministrator(Settings())
        data_reader = CSVDataReader(path_admin)
        self._fake_observation_data = FakeObservationData2D(data_reader, None)
        fake_force_data = FakeForceData2D(data_reader, None)
        fake_solution_data = FakeSolutionData2D(data_reader, None)
        self._spy_data_selection_func = SpyDataSelectionFunc()
        self._num_data_points = 5
        self._num_collocation_points = 4
        self._num_simulation_points_per_edge = 3
        self._PRNG_key = jax.random.PRNGKey(0)
        PRNG_keys = jax.random.split(self._PRNG_key, 2)
        self._PRNG_key_train_data = PRNG_keys[0]
        self._PRNG_key_valid_data = PRNG_keys[1]
        self._sut = Domain2DWithSolution(
            observation_data=self._fake_observation_data,
            force_data=fake_force_data,
            solution_data=fake_solution_data,
            data_selection_func=self._spy_data_selection_func,
            PRNG_key=self._PRNG_key,
        )

    # training data
    def test_select_training_data_number_of_points(self) -> None:
        """
        Test that the number of data and collocation points is passed correctly to the data selection function.
        """
        _ = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = self._spy_data_selection_func.num_points

        expected = [
            self._num_data_points,
            self._num_data_points,
            self._num_collocation_points,
        ]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_select_training_data_PRNG_key(self) -> None:
        """
        Test that the PRNG key for training data is passed correctly to the data selection function.
        """
        _ = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = self._spy_data_selection_func.PRNG_key

        expected = self._PRNG_key_train_data
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)

    # validation data
    def test_select_validation_data_number_of_points(self) -> None:
        """
        Test that the number of validation points is passed correctly to the data selection function.
        """
        _ = self._sut.generate_validation_data(num_data_points=self._num_data_points)
        actual = self._spy_data_selection_func.num_points

        expected = [
            self._num_data_points,
            self._num_data_points,
            self._num_data_points,
            self._num_data_points,
        ]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_select_validation_data_PRNG_key(self) -> None:
        """
        Test that the PRNG key for validation data is passed correctly to the data selection function.
        """
        _ = self._sut.generate_validation_data(num_data_points=self._num_data_points)
        actual = self._spy_data_selection_func.PRNG_key

        expected = self._PRNG_key_valid_data
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)


class TestDomain2DWithoutSolution(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        path_admin = PathAdministrator(Settings())
        data_reader = CSVDataReader(path_admin)
        self._fake_observation_data = FakeObservationData2D(data_reader, None)
        fake_force_data = FakeForceData2D(data_reader, None)
        self._spy_split_data_func = SpyDataSplittingFunc()
        self._spy_data_selection_func = SpyDataSelectionFunc()
        self._num_data_points = 5
        self._num_collocation_points = 4
        self._num_simulation_points_per_edge = 3
        self._PRNG_key = jax.random.PRNGKey(0)
        PRNG_keys = jax.random.split(self._PRNG_key, 3)
        self._PRNG_key_train_data = PRNG_keys[0]
        self._PRNG_key_valid_data = PRNG_keys[1]
        self._PRNG_key_data_splitting = PRNG_keys[2]
        self._sut = Domain2DWithoutSolution(
            observation_data=self._fake_observation_data,
            force_data=fake_force_data,
            proportion_train_data=1.0,
            split_data_func=self._spy_split_data_func,
            data_selection_func=self._spy_data_selection_func,
            PRNG_key=self._PRNG_key,
        )

    # training data
    def test_split_training_data_splitting_PRNG_key(self) -> None:
        """
        Test that the PRNG key for training data splitting is passed correctly to the data splitting function.
        """
        _ = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = self._spy_split_data_func.PRNG_key

        expected = self._PRNG_key_data_splitting
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)

    def test_select_training_data_number_of_points(self) -> None:
        """
        Test that the number of data and collocation points is passed correctly to the data selection function.
        """
        _ = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = self._spy_data_selection_func.num_points

        expected = [
            self._num_data_points,
            self._num_data_points,
            self._num_collocation_points,
        ]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_select_training_data_selection_PRNG_key(self) -> None:
        """
        Test that the PRNG key for training data selection is passed correctly to the data selection function.
        """
        _ = self._sut.generate_training_data(
            num_data_points=self._num_data_points,
            num_collocation_points=self._num_collocation_points,
        )
        actual = self._spy_data_selection_func.PRNG_key

        expected = self._PRNG_key_train_data
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)

    # validation data
    def test_split_validation_data_splitting_PRNG_key(self) -> None:
        """
        Test that the PRNG key for validation data splitting is passed correctly to the data splitting function.
        """
        _ = self._sut.generate_validation_data(num_data_points=self._num_data_points)
        actual = self._spy_split_data_func.PRNG_key

        expected = self._PRNG_key_data_splitting
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)

    def test_select_validation_data_number_of_points(self) -> None:
        """
        Test that the number of validation points is passed correctly to the data selection function.
        """
        _ = self._sut.generate_validation_data(num_data_points=self._num_data_points)
        actual = self._spy_data_selection_func.num_points

        expected = [
            self._num_data_points,
            self._num_data_points,
            self._num_data_points,
            self._num_data_points,
        ]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_select_validation_data_selection_PRNG_key(self) -> None:
        """
        Test that the PRNG key for validation data selection is passed correctly to the data selection function.
        """
        _ = self._sut.generate_validation_data(num_data_points=self._num_data_points)
        actual = self._spy_data_selection_func.PRNG_key

        expected = self._PRNG_key_valid_data
        assert_equal_PRNGKeys(self=self, expected=expected, actual=actual)
