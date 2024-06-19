# Standard library imports
from typing import Protocol
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays, assert_equal
from calibrationpinn.domains import TrainingData1D
from calibrationpinn.models.constantFunction import ConstantFunction
from calibrationpinn.training.pinns.momentumEquation1D import (
    MomentumEquation1DModels,
    momentum_equation_pinn_func,
    traction_func,
    strain_func,
    strain_energy_func,
    youngs_modulus_func,
    first_derivative_displacements_func,
    second_derivative_displacements_func,
    MomentumEquation1DParameters,
    MomentumEquation1DModels,
    MomentumEquation1DEstimates,
)
from calibrationpinn.training.pinns.tests.testdoubles import FakeNetU
from calibrationpinn.typeAliases import HKTransformed, JNPArray


class TestMomentumEquation1DNormalized(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._youngs_modulus = 2.0
        self._youngs_modulus_correction = 0.0
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 1
        net_u = set_up_fake_net_u(self)
        youngs_modulus_correction_func = set_up_fake_youngs_modulus_correction(self)
        params_net_u = net_u.init(PRNG_key, input=jnp.zeros(input_size))
        params_youngs_modulus_correction = youngs_modulus_correction_func.init(
            PRNG_key, input=jnp.zeros(input_size)
        )
        self._parameters = MomentumEquation1DParameters(
            net_u=params_net_u,
            youngs_modulus_correction=params_youngs_modulus_correction,
        )
        self._models = MomentumEquation1DModels(
            net_u=net_u,
            youngs_modulus_correction=youngs_modulus_correction_func,
        )
        youngs_modulus_estimate = self._youngs_modulus
        self._estimates = MomentumEquation1DEstimates(
            youngs_modulus=youngs_modulus_estimate,
        )
        self._sut = momentum_equation_pinn_func

    def test_momentum_equation_pinn_for_positive_inputs(self) -> None:
        """
        Test that the momentum equation pinn function returns zeros for positive inputs.
        """
        single_input = jnp.array([2.0])
        training_data = generate_fake_training_data(self, single_input)

        actual = self._sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_momentum_equation_pinn_for_negative_inputs(self) -> None:
        """
        Test that the momentum equation pinn function returns zeros for positive inputs.
        """
        single_input = jnp.array([-2.0])
        training_data = generate_fake_training_data(self, single_input)

        actual = self._sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_momentum_equation_pinn_for_zero_inputs(self) -> None:
        """
        Test that the momentum equation pinn function returns zeros for positive inputs.
        """
        single_input = jnp.array([0.0])
        training_data = generate_fake_training_data(self, single_input)

        actual = self._sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_traction_func(self) -> None:
        """
        Test that the traction is calculated correctly.
        """
        single_input = jnp.array([2.0])

        actual = traction_func(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            single_input=single_input,
        )

        expected = jnp.array([16.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_func(self) -> None:
        """
        Test that the strain func is calculated correctly.
        """
        single_input = jnp.array([2.0])
        actual = strain_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = jnp.array([8.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_strain_energy_func(self) -> None:
        """
        Test that the strain energy is calculated correctly.
        """
        single_input = jnp.array([2.0])

        actual = strain_energy_func(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            single_input=single_input,
        )
        expected = jnp.array([64.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_youngs_modulus_func(self) -> None:
        """
        Test that the youngs modulus is calculated correctly.
        """
        single_input = jnp.array([2.0])
        actual = youngs_modulus_func(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            single_input=single_input,
        )

        expected = 2.0
        assert_equal(self=self, expected=expected, actual=actual)

    def test_first_derivative_displacements_func(self) -> None:
        """
        Test that the first derivative of the displacements func is calculated correctly.
        """
        single_input = jnp.array([2.0])
        actual = first_derivative_displacements_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = jnp.array([8.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_second_derivative_displacements_func(self) -> None:
        """
        Test that the second derivative of the displacements func is calculated correctly.
        """
        single_input = jnp.array([2.0])
        actual = second_derivative_displacements_func(
            params=self._parameters, models=self._models, single_input=single_input
        )

        expected = jnp.array([4.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)


class TestCaseProtocol(Protocol):
    _youngs_modulus: float
    _youngs_modulus_correction: float
    _parameters: MomentumEquation1DParameters
    _models: MomentumEquation1DModels
    _estimates: MomentumEquation1DEstimates


def generate_fake_training_data(
    test_case: TestCaseProtocol, single_input: JNPArray
) -> TrainingData1D:
    volume_force = jnp.array([-8.0])
    return TrainingData1D(
        x_data=jnp.array([0.0]),
        y_data_true=jnp.array([0.0]),
        x_pde=jnp.array([0.0]),
        y_pde_true=jnp.array([0.0]),
        volume_force=volume_force,
        x_traction_bc=jnp.array([0.0]),
        y_traction_bc_true=jnp.array([0.0]),
    )


def set_up_fake_net_u(test_case: TestCaseProtocol) -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetU()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_youngs_modulus_correction(
    test_case: TestCaseProtocol,
) -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = ConstantFunction(
            output_size=1,
            func_value_init=hk.initializers.Constant(
                test_case._youngs_modulus_correction
            ),
            name="fake_youngs_modulus_correction",
        )
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))
