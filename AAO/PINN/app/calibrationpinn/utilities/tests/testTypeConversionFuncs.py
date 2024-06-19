# Standard library imports
import unittest

# Third-party imports
import jax.numpy as jnp
import numpy as np

# Local library imports
from calibrationpinn.assertions import assert_true
from calibrationpinn.typeAliases import JNPArray, NPArray
from calibrationpinn.utilities.typeConversionFuncs import (
    numpy_to_jax_numpy,
    jax_numpy_to_numpy,
)


class TestNumpyToJAXNumpy(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = numpy_to_jax_numpy

    def test_numpy_array_to_JAX_numpy_array_conversion_func(self) -> None:
        """
        Test that the numpy array is correctly converted to a jax.numpy array.
        """
        np_array: NPArray = np.array([0.0])
        jnp_array = self._sut(np_array)

        assert_true(self=self, expression=isinstance(jnp_array, jnp.ndarray))


class TestJAXNumpyToNumpy(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = jax_numpy_to_numpy

    def test_JAX_numpy_array_to_numpy_array_conversion_func(self) -> None:
        """
        Test that the jax.numpy array is correctly converted to a numpy array.
        """
        jnp_array = jnp.array([0.0])
        np_array = self._sut(jnp_array)

        assert_true(self=self, expression=isinstance(np_array, np.ndarray))
