# Standard library imports
import unittest

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains.splitDataFunc import split_in_train_and_valid_data


class TestSplitInTrainAndValidDataFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._data = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self._PRNG_key = jax.random.PRNGKey(0)
        self._sut = split_in_train_and_valid_data

    def test_fitting_proportion_train_data(self):
        """
        Test that the training data is split correctly for a fitting proportion.
        """
        proportion_train_data = 0.8

        actual, _ = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[:4]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_fitting_proportion_valid_data(self):
        """
        Test that the validation data is split correctly for a fitting proportion.
        """
        proportion_train_data = 0.8

        _, actual = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[4:5]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_unfitting_proportion_train_data(self):
        """
        Test that the training data is split correctly for an unfitting proportion.
        """
        proportion_train_data = 0.75

        actual, _ = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[:3]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_unfitting_proportion_valid_data(self):
        """
        Test that the validation data is split correctly for an unfitting proportion.
        """
        proportion_train_data = 0.75

        _, actual = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        shuffled_data = jax.random.permutation(self._PRNG_key, self._data)
        expected = shuffled_data[3:5]
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_full_proportion_train_data(self):
        """
        Test that the full data set is used as training data for the full proportion.
        """
        proportion_train_data = 1.0

        actual, _ = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        expected = self._data
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_full_proportion_valid_data(self):
        """
        Test that the full data set is used as validation data for the full proportion.
        """
        proportion_train_data = 1.0

        _, actual = self._sut(
            data=self._data,
            proportion_train_data=proportion_train_data,
            PRNG_key=self._PRNG_key,
        )

        expected = self._data
        assert_equal_arrays(self=self, expected=expected, actual=actual)
