# Standard library imports
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.models.layers import ActivatedLayer
from calibrationpinn.typeAliases import JNPArray


class TestActivatedLayer(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        def activation_func(inputs):
            return 2.0 * inputs

        def layer_func(input: JNPArray) -> JNPArray:
            layer = ActivatedLayer(
                output_size=2,
                w_init=hk.initializers.Constant(2.0),
                b_init=hk.initializers.Constant(3.0),
                activation=activation_func,
                name="test_activated_layer",
            )
            return layer(input)

        self._sut = hk.without_apply_rng(hk.transform(layer_func))
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 4
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_activated_layer(self) -> None:
        """
        Test that the activated layer calculates the outputs correctly from the inputs.
        """
        inputs = jnp.array([[0.0, 1.0, 2.0, 3.0], [0.0, -1.0, -2.0, -3.0]])

        actual = self._sut.apply(self._params, inputs)

        expected = jnp.array([[30.0, 30.0], [-18.0, -18.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)
