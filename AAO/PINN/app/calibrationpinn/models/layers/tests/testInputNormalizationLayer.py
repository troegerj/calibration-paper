# Standard library imports
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.models.layers import InputNormalizationLayer
from calibrationpinn.typeAliases import JNPArray


class TestInputNormalizationLayerWithPositiveInputsOnly(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        def layer_func(input: JNPArray) -> JNPArray:
            layer = InputNormalizationLayer(
                min_input=jnp.array([0.0, 0.0]),
                max_input=jnp.array([10.0, 0.1]),
                name="test_input_normalization_layer",
            )
            return layer(input)

        self._sut = hk.without_apply_rng(hk.transform(layer_func))
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_input_normalization_layer(self) -> None:
        """
        Test that the inputs are normalized correctly.
        """
        inputs = jnp.array([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]])

        actual = self._sut.apply(self._params, inputs)

        expected = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestInputNormalizationLayerWithNegativeInputsOnly(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        def layer_func(input: JNPArray) -> JNPArray:
            layer = InputNormalizationLayer(
                min_input=jnp.array([-10.0, -0.1]),
                max_input=jnp.array([0.0, 0.0]),
                name="test_input_normalization_layer",
            )
            return layer(input)

        self._sut = hk.without_apply_rng(hk.transform(layer_func))
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_input_normalization_layer(self) -> None:
        """
        Test that the inputs are normalized correctly.
        """
        inputs = jnp.array([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]])

        actual = self._sut.apply(self._params, inputs)

        expected = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestInputNormalizationLayerWithPositiveAndNegativeInputs(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        def layer_func(input: JNPArray) -> JNPArray:
            layer = InputNormalizationLayer(
                min_input=jnp.array([-10.0, -0.1]),
                max_input=jnp.array([10.0, 0.1]),
                name="test_input_normalization_layer",
            )
            return layer(input)

        self._sut = hk.without_apply_rng(hk.transform(layer_func))
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        self._params = self._sut.init(PRNG_key, input=jnp.zeros(input_size))

    def test_input_normalization_layer(self) -> None:
        """
        Test that the inputs are normalized correctly.
        """
        inputs = jnp.array([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]])

        actual = self._sut.apply(self._params, inputs)

        expected = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)
