# Standard library imports
import unittest

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains.dataselection import (
    select_data_randomly,
    select_data_sequentially,
)


class TestRandomDataSelectionFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = select_data_randomly
        self._data = jnp.array([0.0, 1.0, 2.0, 3.0]).reshape((-1, 1))
        self._PRNG_key = jax.random.PRNGKey(0)

    def test_random_data_selection_func_when_not_all_data_should_be_selected(
        self,
    ) -> None:
        """
        Test that the data is shuffled and the specified number of data points is returned.
        """
        num_data_points = 3
        actual = self._sut(self._data, num_data_points, self._PRNG_key)

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[:num_data_points]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_random_data_selection_func_when_all_data_should_be_selected(self) -> None:
        """
        Test that the data is shuffled and the specified number of data points is returned.
        """
        num_data_points = 4
        actual = self._sut(self._data, num_data_points, self._PRNG_key)

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[:num_data_points]
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestSequentialDataSelectionFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = select_data_sequentially
        self._data = jnp.array([0.0, 1.0, 2.0, 3.0]).reshape((-1, 1))
        self._PRNG_key = jax.random.PRNGKey(0)

    def test_sequential_data_selection_func_when_not_all_data_should_be_selected(
        self,
    ) -> None:
        """
        Test that the data is sequentially ordered and the specified number of data points is returned.
        """
        num_data_points = 3
        actual = self._sut(self._data, num_data_points, self._PRNG_key)

        expected = self._data[:num_data_points]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_sequential_data_selection_func_when_all_data_should_be_selected(
        self,
    ) -> None:
        """
        Test that the data is sequentially ordered and the specified number of data points is returned.
        """
        num_data_points = 4
        actual = self._sut(self._data, num_data_points, self._PRNG_key)

        expected = self._data[:num_data_points]
        assert_equal_arrays(self=self, expected=expected, actual=actual)
