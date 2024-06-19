# Standard library imports
from functools import partial
from typing import Any, Callable, TypeAlias

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray, JNPFloat
from calibrationpinn.domains import TrainingData2D


Estimates: TypeAlias = Any
Models: TypeAlias = Any
Parameters: TypeAlias = Any


StressFunc: TypeAlias = Callable[[Parameters, Models, Estimates, JNPArray], JNPArray]
PINNFunc: TypeAlias = Callable[
    [Parameters, Models, Estimates, TrainingData2D, JNPArray], JNPArray
]
StrainEnergyFunc: TypeAlias = Callable[
    [Parameters, Models, Estimates, JNPArray], JNPArray
]
TractionFunc: TypeAlias = Callable[
    [Parameters, Models, Estimates, JNPArray, JNPArray], JNPArray
]


def _pinn_func(voigt_stress_func: StressFunc):
    @partial(jax.jit, static_argnums=(1,))
    def _momentum_equation_pinn_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        train_data: TrainingData2D,
        single_input: JNPArray,
    ) -> JNPArray:
        return (
            _divergence_stress_func(
                params,
                models,
                estimates,
                single_input,
                voigt_stress_func,
            )
            + train_data.volume_force
        )

    return _momentum_equation_pinn_func


def _strain_energy_func(stress_func: StressFunc):
    @partial(jax.jit, static_argnums=(1,))
    def _strain_energy_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        single_input: JNPArray,
    ) -> JNPArray:
        stress = stress_func(
            params,
            models,
            estimates,
            single_input,
        )
        strain = strain_func(params, models, single_input)
        return (1 / 2) * jnp.einsum("ij,ij", stress, strain)

    return _strain_energy_func


def _traction_func(stress_func: StressFunc):
    @partial(jax.jit, static_argnums=(1,))
    def _traction_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        normal_vector: JNPArray,
        single_input: JNPArray,
    ) -> JNPArray:
        stress = stress_func(
            params,
            models,
            estimates,
            single_input,
        )
        return jnp.matmul(stress, normal_vector)

    return _traction_func


def _divergence_stress_func(
    params: Parameters,
    models: Models,
    estimates: Estimates,
    single_input: JNPArray,
    voigt_stress_func: StressFunc,
) -> JNPArray:
    jac_sigma = _jacobian_voigt_stress_func(
        params, models, estimates, single_input, voigt_stress_func
    )
    dSigmaXX_dX = jac_sigma[0, 0]
    dSigmaXY_dY = jac_sigma[2, 1]
    dSigmaYX_dX = jac_sigma[2, 0]
    dSigmaYY_dY = jac_sigma[1, 1]
    return jnp.array([dSigmaXX_dX + dSigmaXY_dY, dSigmaYX_dX + dSigmaYY_dY])


def _jacobian_voigt_stress_func(
    params: Parameters,
    models: Models,
    estimates: Estimates,
    single_input: JNPArray,
    voigt_stress_func: StressFunc,
) -> JNPArray:
    return jax.jacrev(voigt_stress_func, 3)(params, models, estimates, single_input)


@partial(jax.jit, static_argnums=(1,))
def strain_func(params: Parameters, models: Models, single_input: JNPArray) -> JNPArray:
    jac_u = jacobian_displacements_func(params, models, single_input)
    return 1 / 2 * (jac_u + jnp.transpose(jac_u))


def jacobian_displacements_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    return jax.jacrev(displacements_func, argnums=2)(params, models, single_input)


@partial(jax.jit, static_argnums=(1,))
def displacements_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    u_x = models.net_ux.apply(params.net_ux, single_input)[0]
    u_y = models.net_uy.apply(params.net_uy, single_input)[0]
    return jnp.asarray([u_x, u_y])


def material_parameter_func(correction: float, estimate: float) -> JNPFloat:
    return (1.0 + correction) * estimate
