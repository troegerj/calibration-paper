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


class MomentumEquation2DEstimates_E_nu(NamedTuple):
    youngs_modulus: float
    poissons_ratio: float


class MomentumEquation2DModels_E_nu(NamedTuple):
    net_ux: HKTransformed
    net_uy: HKTransformed
    youngs_modulus_correction: HKTransformed
    poissons_ratio_correction: HKTransformed


class MomentumEquation2DParameters_E_nu(NamedTuple):
    net_ux: JNPPyTree
    net_uy: JNPPyTree
    youngs_modulus_correction: JNPPyTree
    poissons_ratio_correction: JNPPyTree


Parameters: TypeAlias = MomentumEquation2DParameters_E_nu
Models: TypeAlias = MomentumEquation2DModels_E_nu
Estimates: TypeAlias = MomentumEquation2DEstimates_E_nu


def pinn_func_factory(state: str) -> PINNFunc:
    if state == "plane strain":
        voigt_stress_func = _voigt_stress_func(
            _voigt_material_parameters_tensor_func_plane_strain
        )
    elif state == "plane stress":
        voigt_stress_func = _voigt_stress_func(
            _voigt_material_parameters_tensor_func_plane_stress
        )
    return _pinn_func(voigt_stress_func)


def strain_energy_func_factory(state: str) -> StrainEnergyFunc:
    if state == "plane strain":
        stress_func = _stress_func(_voigt_material_parameters_tensor_func_plane_strain)
    elif state == "plane stress":
        stress_func = _stress_func(_voigt_material_parameters_tensor_func_plane_stress)
    return _strain_energy_func(stress_func)


def traction_func_factory(state: str) -> TractionFunc:
    if state == "plane strain":
        stress_func = _stress_func(_voigt_material_parameters_tensor_func_plane_strain)
    elif state == "plane stress":
        stress_func = _stress_func(_voigt_material_parameters_tensor_func_plane_stress)
    return _traction_func(stress_func)


def _voigt_material_parameters_tensor_func_plane_strain(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    E_cor = models.youngs_modulus_correction.apply(
        params.youngs_modulus_correction, single_input
    )[0]
    nu_cor = models.poissons_ratio_correction.apply(
        params.poissons_ratio_correction, single_input
    )[0]
    E = material_parameter_func(E_cor, estimates.youngs_modulus)
    nu = material_parameter_func(nu_cor, estimates.poissons_ratio)
    lambda_ = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_ = E / (2.0 * (1.0 + nu))
    return jnp.array(
        [
            [lambda_ + 2.0 * mu_, lambda_, 0.0],
            [lambda_, lambda_ + 2.0 * mu_, 0.0],
            [0.0, 0.0, mu_],
        ]
    )


def _voigt_material_parameters_tensor_func_plane_stress(
    params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
) -> JNPArray:
    E_cor = models.youngs_modulus_correction.apply(
        params.youngs_modulus_correction, single_input
    )[0]
    nu_cor = models.poissons_ratio_correction.apply(
        params.poissons_ratio_correction, single_input
    )[0]
    E = material_parameter_func(E_cor, estimates.youngs_modulus)
    nu = material_parameter_func(nu_cor, estimates.poissons_ratio)
    return (E / (1.0 - nu**2)) * jnp.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )


def _stress_func(voigt_material_parameters_tensor_func) -> StressFunc:
    voigt_stress_func = _voigt_stress_func(voigt_material_parameters_tensor_func)

    def stress_func(
        params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
    ) -> JNPArray:
        voigt_stress = voigt_stress_func(
            params,
            models,
            estimates,
            single_input,
        )
        stress = jnp.array(
            [[voigt_stress[0], voigt_stress[2]], [voigt_stress[2], voigt_stress[1]]]
        )
        return stress

    return stress_func


def _voigt_stress_func(
    voigt_material_parameters_tensor_func,
) -> StressFunc:
    def voigt_stress_func(
        params: Parameters, models: Models, estimates: Estimates, single_input: JNPArray
    ) -> JNPArray:
        voigt_material_params = voigt_material_parameters_tensor_func(
            params, models, estimates, single_input
        )
        voigt_strain = _voigt_strain_func(params, models, single_input)
        return jnp.matmul(voigt_material_params, voigt_strain)

    return voigt_stress_func


def _voigt_strain_func(
    params: Parameters, models: Models, single_input: JNPArray
) -> JNPArray:
    strain = strain_func(params, models, single_input)
    e_xx = strain[0, 0]
    e_yy = strain[1, 1]
    e_xy = strain[0, 1]
    return jnp.asarray([e_xx, e_yy, 2 * e_xy])
