# Standard library imports
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.models import ModelBuilder


class TestConstantFunctionWithOutputSizeOne(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        model_builder = ModelBuilder()
        self._constant_func_value = 2.0
        self._sut = model_builder.build_constant_function(
            output_size=1,
            func_value_init=hk.initializers.Constant(self._constant_func_value),
            name="test_constant_function",
        )
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_constant_function_with_single_input(self) -> None:
        """
        Test that the constant function returns the correct output for a single input.
        """
        input = jnp.array([-1.0, 1.0])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array([self._constant_func_value])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_constant_function_with_multiple_inputs(self) -> None:
        """
        Test that the constant function returns the correct outputs for multiple inputs.
        """
        input = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array([[self._constant_func_value], [self._constant_func_value]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestConstantFunctionWithOutputSizeTwo(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        model_builder = ModelBuilder()
        self._constant_func_value = jnp.array([2.0, 3.0])
        self._sut = model_builder.build_constant_function(
            output_size=2,
            func_value_init=hk.initializers.Constant(self._constant_func_value),
            name="test_constant_function",
        )
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_constant_function_with_single_input(self) -> None:
        """
        Test that the constant function returns the correct output for a single input.
        """
        input = jnp.array([-1.0, 1.0])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array(self._constant_func_value)
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_constant_function_with_multiple_inputs(self) -> None:
        """
        Test that the constant function returns the correct outputs for multiple inputs.
        """
        input = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array([self._constant_func_value, self._constant_func_value])
        assert_equal_arrays(self=self, expected=expected, actual=actual)
