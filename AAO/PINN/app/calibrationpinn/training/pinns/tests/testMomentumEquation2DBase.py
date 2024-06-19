# Standard library imports
from typing import NamedTuple
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal, assert_equal_arrays
from calibrationpinn.training.pinns.momentumEquation2DBase import (
    displacements_func,
    jacobian_displacements_func,
    material_parameter_func,
    strain_func,
)
from calibrationpinn.training.pinns.tests.testdoubles import FakeNetUx, FakeNetUy
from calibrationpinn.typeAliases import HKTransformed, JNPArray, JNPPyTree


class FakeMomentumEquation2DModels(NamedTuple):
    net_ux: HKTransformed
    net_uy: HKTransformed


class FakeMomentumEquation2DParameters(NamedTuple):
    net_ux: JNPPyTree
    net_uy: JNPPyTree


class TestMomentumEquation2DBase(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._material_parameter = 2.0
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        net_ux = set_up_fake_net_ux()
        net_uy = set_up_fake_net_uy()
        self._material_parameter_correction = 3.0
        params_net_ux = net_ux.init(PRNG_key, input=jnp.zeros(input_size))
        params_net_uy = net_uy.init(PRNG_key, input=jnp.zeros(input_size))
        self._parameters = FakeMomentumEquation2DParameters(
            net_ux=params_net_ux,
            net_uy=params_net_uy,
        )
        self._models = FakeMomentumEquation2DModels(
            net_ux=net_ux,
            net_uy=net_uy,
        )

    def test_displacements_func(self) -> None:
        """
        Test that the displacements are calculated correctly.
        """
        single_input = jnp.array([1.0, 3.0])
        actual = displacements_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = jnp.array([1.5, 9.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_jacobian_displacements_func(self) -> None:
        """
        Test that the jacobian of the displacements func is calculated correctly.
        """
        single_input = jnp.array([1.0, 3.0])
        actual = jacobian_displacements_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = jnp.array([[3.0, 0.5], [9.0, 6.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_strain_func(self) -> None:
        """
        Test that the strain is calculated correctly.
        """
        single_input = jnp.array([1.0, 3.0])
        actual = strain_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = (1 / 2) * jnp.array(
            [
                [3.0 + 3.0, 0.5 + 9.0],
                [9.0 + 0.5, 6.0 + 6.0],
            ]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_material_parameter_func(self) -> None:
        """
        Test that the material parameter is calculated correctly.
        """

        actual = material_parameter_func(
            correction=self._material_parameter_correction,
            estimate=self._material_parameter,
        )

        expected = (
            1.0 + self._material_parameter_correction
        ) * self._material_parameter
        assert_equal(self=self, expected=expected, actual=actual)


def set_up_fake_net_ux() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUx()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_net_uy() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUy()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))
