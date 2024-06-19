# Standard library imports
from typing import Callable, NamedTuple, TypeAlias

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.training.pinns.momentumEquation2DBase import (
    displacements_func,
    material_parameter_func,
    strain_func,
    _pinn_func,
    _strain_energy_func,
    _traction_func,
    PINNFunc,
    StrainEnergyFunc,
    StressFunc,
    TractionFunc,
)
from calibrationpinn.typeAliases import HKTransformed, JNPArray, JNPPyTree


class MomentumEquation2DEstimates_K_G(NamedTuple):
    bulk_modulus: float
    shear_modulus: float


class MomentumEquation2DModels_K_G(NamedTuple):
    net_ux: HKTransformed
    net_uy: HKTransformed
    bulk_modulus_correction: HKTransformed
    shear_modulus_correction: HKTransformed


class MomentumEquation2DParameters_K_G(NamedTuple):
    net_ux: JNPPyTree
    net_uy: JNPPyTree
    bulk_modulus_correction: JNPPyTree
    shear_modulus_correction: JNPPyTree


Parameters: TypeAlias = MomentumEquation2DParameters_K_G
Models: TypeAlias = MomentumEquation2DModels_K_G
Estimates: TypeAlias = MomentumEquation2DEstimates_K_G


def calculate_K_from_E_and_nu_factory(elasticity_state: str):
    if elasticity_state == "plane strain":
        return _calculate_K_from_E_and_nu_plane_strain
    elif elasticity_state == "plane stress":
        return _calculate_K_from_E_and_nu_plane_stress


def _calculate_K_from_E_and_nu_plane_strain(E: float, nu: float):
    return E / (2 * (1 + nu) * (1 - 2 * nu))


def _calculate_K_from_E_and_nu_plane_stress(E: float, nu: float):
    return E / (3 * (1 - 2 * nu))


def calculate_G_from_E_and_nu(E: float, nu: float):
    return E / (2 * (1 + nu))


def calculate_E_from_K_and_G_factory(elasticity_state: str):
    if elasticity_state == "plane strain":
        return calculate_E_from_K_and_G_plane_strain
    elif elasticity_state == "plane stress":
        return calculate_E_from_K_and_G_plane_stress


def calculate_E_from_K_and_G_plane_strain(K: float, G: float):
    return G * (-G / K + 3)


def calculate_E_from_K_and_G_plane_stress(K: float, G: float):
    return (9 * K * G) / (3 * K + G)


def calculate_nu_from_K_and_G_factory(elasticity_state: str):
    if elasticity_state == "plane strain":
        return calculate_nu_from_K_and_G_plane_strain
    elif elasticity_state == "plane stress":
        return calculate_nu_from_K_and_G_plane_stress


def calculate_nu_from_K_and_G_plane_strain(K: float, G: float):
    return -G / (2 * K) + 0.5


def calculate_nu_from_K_and_G_plane_stress(K: float, G: float):
    return (3 * K - 2 * G) / (6 * K + 2 * G)


def pinn_func_factory(state: str) -> PINNFunc:
    if state == "plane strain":
        voigt_stress_func = _voigt_stress_func(_stress_func_plane_strain)
    elif state == "plane stress":
        voigt_stress_func = _voigt_stress_func(_stress_func_plane_stress)
    return _pinn_func(voigt_stress_func)


def strain_energy_func_factory(state: str) -> StrainEnergyFunc:
    if state == "plane strain":
        stress_func = _stress_func_plane_strain
    elif state == "plane stress":
        stress_func = _stress_func_plane_stress
    return _strain_energy_func(stress_func)


def traction_func_factory(state: str) -> TractionFunc:
    if state == "plane strain":
        stress_func = _stress_func_plane_strain
    elif state == "plane stress":
        stress_func = _stress_func_plane_stress
    return _traction_func(stress_func)


def _stress_func_plane_strain(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    K_cor = models.bulk_modulus_correction.apply(
        params.bulk_modulus_correction, single_input
    )[0]
    G_cor = models.shear_modulus_correction.apply(
        params.shear_modulus_correction, single_input
    )[0]
    K = material_parameter_func(K_cor, estimates.bulk_modulus)
    G = material_parameter_func(G_cor, estimates.shear_modulus)
    volumetric_strain = _volumetric_strain_func(params, models, single_input)
    deviatoric_strain = _deviatoric_strain_func(params, models, single_input)
    return K * volumetric_strain + 2 * G * (deviatoric_strain)


def _volumetric_strain_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    strain = strain_func(params, models, single_input)
    trace_strain = jnp.trace(strain)
    identity = jnp.identity(2)
    return trace_strain * identity


def _deviatoric_strain_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    strain = strain_func(params, models, single_input)
    volumetric_strain = _volumetric_strain_func(params, models, single_input)
    return strain - (volumetric_strain / 2)


def _stress_func_plane_stress(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    K_cor = models.bulk_modulus_correction.apply(
        params.bulk_modulus_correction, single_input
    )[0]
    G_cor = models.shear_modulus_correction.apply(
        params.shear_modulus_correction, single_input
    )[0]
    K = material_parameter_func(K_cor, estimates.bulk_modulus)
    G = material_parameter_func(G_cor, estimates.shear_modulus)
    strain = strain_func(params, models, single_input)
    eps_xx = strain[0, 0]
    eps_yy = strain[1, 1]
    eps_xy = strain[0, 1]
    sig_xx = (2 * G / (3 * K + 4 * G)) * (
        (6 * K + 2 * G) * eps_xx + (3 * K - 2 * G) * eps_yy
    )
    sig_yy = (2 * G / (3 * K + 4 * G)) * (
        (3 * K - 2 * G) * eps_xx + (6 * K + 2 * G) * eps_yy
    )
    sig_xy = G * 2 * eps_xy
    return jnp.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])


# def _stress_func_plane_stress(
#     params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
# ) -> JNPArray:
#     K_cor = models.bulk_modulus_correction.apply(
#         params.bulk_modulus_correction, single_input
#     )[0]
#     G_cor = models.shear_modulus_correction.apply(
#         params.shear_modulus_correction, single_input
#     )[0]
#     K = material_parameter_func(K_cor, estimates.bulk_modulus)
#     G = material_parameter_func(G_cor, estimates.shear_modulus)
#     deviatoric_strain_func = _deviatoric_strain_func(denominator=2)
#     volumetric_strain = _volumetric_strain_func(params, models, single_input)
#     deviatoric_strain = deviatoric_strain_func(params, models, single_input)
#     return K * volumetric_strain + 2 * G * (deviatoric_strain)


def _voigt_stress_func(
    stress_func: StressFunc,
) -> StressFunc:
    def voigt_stress_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        single_input: JNPArray,
    ) -> JNPArray:
        stress = stress_func(params, models, estimates, single_input)
        return jnp.array([stress[0, 0], stress[1, 1], stress[0, 1]])

    return voigt_stress_func
