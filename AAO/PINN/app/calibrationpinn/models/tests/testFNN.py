# Standard library imports
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.models import ModelBuilder


class TestFNN(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        def activation_func(inputs):
            return 2.0 * inputs

        model_builder = ModelBuilder()
        self._sut = model_builder.build_feedforward_neural_network(
            output_sizes=[2, 2, 2],
            w_init=hk.initializers.Constant(1.0),
            b_init=hk.initializers.Constant(2.0),
            activation=activation_func,
            name="test_fnn",
        )
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_FNN_single_input(self) -> None:
        """
        Test that the forward neural network returns the correct output for a single input.
        """
        input = jnp.array([1.0, 2.0])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array([90.0, 90.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_FNN_multiple_inputs(self) -> None:
        """
        Test that the forward neural network returns the correct outputs for multiple inputs.
        """
        input = jnp.array([[1.0, 2.0], [1.0, 2.0]])

        actual = self._sut.apply(self._params, input)

        expected = jnp.array([[90.0, 90.0], [90.0, 90.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)
