# Standard library imports
from functools import partial
from typing import Callable, TypeAlias, NamedTuple

# Third-party imports
import jax

# Local library imports
from calibrationpinn.typeAliases import HKTransformed, JNPArray, JNPFloat, JNPPyTree
from calibrationpinn.domains import TrainingData1D


class MomentumEquation1DParameters(NamedTuple):
    net_u: JNPPyTree
    youngs_modulus_correction: JNPPyTree


class MomentumEquation1DModels(NamedTuple):
    net_u: HKTransformed
    youngs_modulus_correction: HKTransformed


class MomentumEquation1DEstimates(NamedTuple):
    youngs_modulus: float


Parameters: TypeAlias = MomentumEquation1DParameters
Models: TypeAlias = MomentumEquation1DModels
Estimates: TypeAlias = MomentumEquation1DEstimates


@partial(jax.jit, static_argnums=(1,))
def momentum_equation_pinn_func(
    params: Parameters,
    models: Models,
    estimates: Estimates,
    train_data: TrainingData1D,
    single_input: JNPArray,
) -> JNPArray:
    return (
        youngs_modulus_func(params, models, estimates, single_input)
        * second_derivative_displacements_func(params, models, single_input)
        + train_data.volume_force
    )


@partial(jax.jit, static_argnums=(1,))
def traction_func(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    return first_derivative_displacements_func(
        params, models, single_input
    ) * youngs_modulus_func(params, models, estimates, single_input)


@partial(jax.jit, static_argnums=(1,))
def strain_func(params: Parameters, models: Models, single_input: JNPArray) -> JNPArray:
    return first_derivative_displacements_func(params, models, single_input)


@partial(jax.jit, static_argnums=(1,))
def strain_energy_func(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    return (
        (1 / 2)
        * youngs_modulus_func(params, models, estimates, single_input)
        * (first_derivative_displacements_func(params, models, single_input) ** 2)
    )


def youngs_modulus_func(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPFloat:
    return (
        1.0
        + models.youngs_modulus_correction.apply(
            params.youngs_modulus_correction, single_input
        )[0]
    ) * estimates.youngs_modulus


def first_derivative_displacements_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    def apply_net(params: Parameters, single_input: JNPArray) -> JNPFloat:
        return models.net_u.apply(params.net_u, single_input)[0]

    def grad_net(
        params: Parameters,
        apply_net: Callable[[Parameters, JNPArray], JNPArray],
        single_input: JNPFloat,
    ) -> JNPArray:
        return jax.grad(apply_net, 1)(params, single_input)

    return grad_net(params, apply_net, single_input)


def second_derivative_displacements_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    def apply_net(params: Parameters, single_input: JNPArray) -> JNPFloat:
        return models.net_u.apply(params.net_u, single_input)[0]

    def grad_net(
        params: Parameters,
        apply_net: Callable[[Parameters, JNPArray], JNPArray],
        single_input: JNPArray,
    ) -> JNPFloat:
        return jax.grad(apply_net, 1)(params, single_input)[0]

    def grad_grad_net(
        params: Parameters,
        apply_net: Callable[[Parameters, JNPArray], JNPArray],
        single_input: JNPArray,
    ) -> JNPArray:
        return jax.grad(grad_net, 2)(params, apply_net, single_input)

    return grad_grad_net(params, apply_net, single_input)
