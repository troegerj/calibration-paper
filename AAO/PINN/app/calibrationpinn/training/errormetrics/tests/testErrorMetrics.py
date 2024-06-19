# Standard library imports
import unittest

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal
from calibrationpinn.training.errormetrics import (
    mean_squared_error,
    relative_mean_squared_error,
    mean_absolute_error,
    relative_error,
    l2_norm,
    relative_l2_norm,
)


class TestMeanSquaredError(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = mean_squared_error

    def test_mean_squared_error_1D(self) -> None:
        """
        Test that the mean squared error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0], [1.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = ((2.0) ** 2 + (1.0) ** 2) / 2.0
        assert_equal(self=self, expected=expected, actual=actual)

    def test_mean_squared_error_2D(self) -> None:
        """
        Test that the mean squared error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = ((2.0) ** 2 + (2.0) ** 2 + (1.0) ** 2 + (1.0) ** 2) / 4.0
        assert_equal(self=self, expected=expected, actual=actual)


class TestRelativeMeanSquaredError(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._characteristic_length = 2.0
        self._sut = relative_mean_squared_error(
            characteristic_length=self._characteristic_length
        )

    def test_relative_mean_squared_error_1D(self) -> None:
        """
        Test that the relative mean squared error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0], [1.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (
            (2.0 / self._characteristic_length) ** 2
            + (1.0 / self._characteristic_length) ** 2
        ) / 2.0
        assert_equal(self=self, expected=expected, actual=actual)

    def test_relative_mean_squared_error_2D(self) -> None:
        """
        Test that the mean squared error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (
            (2.0 / self._characteristic_length) ** 2
            + (2.0 / self._characteristic_length) ** 2
            + (1.0 / self._characteristic_length) ** 2
            + (1.0 / self._characteristic_length) ** 2
        ) / 4.0
        assert_equal(self=self, expected=expected, actual=actual)


class TestMeanAbsoluteError(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = mean_absolute_error

    def test_mean_absolute_error_1D(self) -> None:
        """
        Test that the mean absolute error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0], [1.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (2.0 + 1.0) / 2.0
        assert_equal(self=self, expected=expected, actual=actual)

    def test_mean_absolute_error_2D(self) -> None:
        """
        Test that the mean absolute error is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (2.0 + 2.0 + 1.0 + 1.0) / 4.0
        assert_equal(self=self, expected=expected, actual=actual)


class TestRelativeError(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = relative_error

    def test_relative_error_1D(self) -> None:
        """
        Test that the relative error is calculated correctly.
        """
        fake_y_true = jnp.array([[1.0], [2.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (1.0 + 0.0) / 2.0
        assert_equal(self=self, expected=expected, actual=actual)

    def test_relative_error_2D(self) -> None:
        """
        Test that the relative error is calculated correctly.
        """
        fake_y_true = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = (1.0 + 1.0 + 0.0 + 0.0) / 4.0
        assert_equal(self=self, expected=expected, actual=actual)


class TestL2Norm(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = l2_norm

    def test_l2_norm_1D(self) -> None:
        """
        Test that the l2-norm is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0], [1.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = jnp.sqrt(2.0**2 + 1.0**2)
        assert_equal(self=self, expected=expected, actual=actual)

    def test_l2_norm_2D(self) -> None:
        """
        Test that the l2-norm is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = jnp.sqrt(2.0**2 + 2.0**2 + 1.0**2 + 1.0**2)
        assert_equal(self=self, expected=expected, actual=actual)


class TestRelativeL2Norm(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = relative_l2_norm

    def test_relative_l2_norm_1D(self) -> None:
        """
        Test that the relative l2-norm is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0], [1.0]])
        fake_y_pred = jnp.array([[2.0], [2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = jnp.sqrt(2.0**2 + 1.0**2) / jnp.sqrt(0.0**2 + 1.0**2)
        assert_equal(self=self, expected=expected, actual=actual)

    def test_relative_l2_norm_2D(self) -> None:
        """
        Test that the relative l2-norm is calculated correctly.
        """
        fake_y_true = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        fake_y_pred = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        actual = self._sut(y_true=fake_y_true, y_pred=fake_y_pred)

        expected = jnp.sqrt(2.0**2 + 2.0**2 + 1.0**2 + 1.0**2) / jnp.sqrt(
            0.0**2 + 0.0**2 + 1.0**2 + 1.0**2
        )
        assert_equal(self=self, expected=expected, actual=actual)
