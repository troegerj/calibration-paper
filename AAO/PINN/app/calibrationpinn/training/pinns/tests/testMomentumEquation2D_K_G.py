# Standard library imports
from typing import Protocol
import unittest

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_almost_equal, assert_equal_arrays
from calibrationpinn.domains.domain2D import TrainingData2D
from calibrationpinn.models.constantFunction import ConstantFunction
from calibrationpinn.training.pinns.momentumEquation2D_K_G import (
    calculate_E_from_K_and_G_factory,
    calculate_nu_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    pinn_func_factory,
    strain_energy_func_factory,
    traction_func_factory,
    MomentumEquation2DParameters_K_G,
    MomentumEquation2DModels_K_G,
    MomentumEquation2DEstimates_K_G,
)
from calibrationpinn.training.pinns.tests.testdoubles import (
    FakeNetUxPlaneStrain,
    FakeNetUyPlaneStrain,
    FakeNetUxPlaneStress,
    FakeNetUyPlaneStress,
)
from calibrationpinn.typeAliases import HKTransformed, JNPArray


class TestMomentumEquation2D_K_G_planeStrain(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._elasticity_state = "plane strain"
        self._youngs_modulus = 1.0
        self._poissons_ratio = 0.25
        calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(
            self._elasticity_state
        )
        self._bulk_modulus = calculate_K_from_E_and_nu(
            self._youngs_modulus, self._poissons_ratio
        )
        self._shear_modulus = calculate_G_from_E_and_nu(
            self._youngs_modulus, self._poissons_ratio
        )
        self._factor_volume_force = 0.0
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        net_ux = set_up_fake_net_ux_for_plane_strain()
        net_uy = set_up_fake_net_uy_for_plane_strain()
        self._bulk_modulus_correction = 0.0
        self._shear_modulus_correction = 0.0
        bulk_modulus_correction_func = set_up_fake_bulk_modulus_correction(self)
        shear_modulus_correction_func = set_up_fake_shear_modulus_correction(self)
        params_net_ux = net_ux.init(PRNG_key, input=jnp.zeros(input_size))
        params_net_uy = net_uy.init(PRNG_key, input=jnp.zeros(input_size))
        params_bulk_modulus_correction = bulk_modulus_correction_func.init(
            PRNG_key, input=jnp.zeros(input_size)
        )
        params_shear_modulus_correction = shear_modulus_correction_func.init(
            PRNG_key, input=jnp.zeros(input_size)
        )
        self._parameters = MomentumEquation2DParameters_K_G(
            net_ux=params_net_ux,
            net_uy=params_net_uy,
            bulk_modulus_correction=params_bulk_modulus_correction,
            shear_modulus_correction=params_shear_modulus_correction,
        )
        self._models = MomentumEquation2DModels_K_G(
            net_ux=net_ux,
            net_uy=net_uy,
            bulk_modulus_correction=bulk_modulus_correction_func,
            shear_modulus_correction=shear_modulus_correction_func,
        )
        self._estimates = MomentumEquation2DEstimates_K_G(
            bulk_modulus=self._bulk_modulus,
            shear_modulus=self._shear_modulus,
        )

    def test_pinn_func_for_positive_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for positive inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([2.0, 2.0])
        training_data = generate_fake_training_data_for_plane_strain(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_pinn_func_for_negative_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for negative inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([-2.0, -2.0])
        training_data = generate_fake_training_data_for_plane_strain(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_pinn_func_for_zero_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for zero inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([0.0, 0.0])
        training_data = generate_fake_training_data_for_plane_strain(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_traction_func(self) -> None:
        """
        Test that the traction is calculated correctly.
        """
        sut = traction_func_factory(self._elasticity_state)
        single_input = jnp.array([1.0, 2.0])
        normal_vector = jnp.array([0.0, 1.0])
        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            normal_vector=normal_vector,
            single_input=single_input,
        )
        e_xx = (1 / 2) * ((4 / 10) + (4 / 10))
        e_yy = (1 / 2) * ((-8 / 5) + (-8 / 5))
        e_xy = (1 / 2) * ((1 / 10) + (-8 / 5))
        lambda_ = (self._youngs_modulus * self._poissons_ratio) / (
            (1 + self._poissons_ratio) * (1 - 2 * self._poissons_ratio)
        )
        mu_ = self._youngs_modulus / (2 * (1 + self._poissons_ratio))
        voigt_sigma = jnp.array(
            [
                (lambda_ + 2 * mu_) * e_xx + lambda_ * e_yy,
                lambda_ * e_xx + (lambda_ + 2 * mu_) * e_yy,
                mu_ * 2 * e_xy,
            ]
        )
        sigma = jnp.array(
            [[voigt_sigma[0], voigt_sigma[2]], [voigt_sigma[2], voigt_sigma[1]]]
        )
        expected = jnp.matmul(sigma, normal_vector)
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_strain_energy_func(self) -> None:
        """
        Test that the strain energy is calculated correctly.
        """
        sut = strain_energy_func_factory(self._elasticity_state)
        single_input = jnp.array([1.0, 2.0])
        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            single_input=single_input,
        )

        e_xx = (1 / 2) * ((4 / 10) + (4 / 10))
        e_yy = (1 / 2) * ((-8 / 5) + (-8 / 5))
        e_xy = (1 / 2) * ((1 / 10) + (-8 / 5))
        epsilon = jnp.array([[e_xx, e_xy], [e_xy, e_yy]])
        lambda_ = (self._youngs_modulus * self._poissons_ratio) / (
            (1 + self._poissons_ratio) * (1 - 2 * self._poissons_ratio)
        )
        mu_ = self._youngs_modulus / (2 * (1 + self._poissons_ratio))
        voigt_sigma = jnp.array(
            [
                (lambda_ + 2 * mu_) * e_xx + lambda_ * e_yy,
                lambda_ * e_xx + (lambda_ + 2 * mu_) * e_yy,
                mu_ * 2 * e_xy,
            ]
        )
        sigma = jnp.array(
            [[voigt_sigma[0], voigt_sigma[2]], [voigt_sigma[2], voigt_sigma[1]]]
        )
        expected = (1 / 2) * (
            sigma[0, 0] * epsilon[0, 0]
            + sigma[0, 1] * epsilon[0, 1]
            + sigma[1, 0] * epsilon[1, 0]
            + sigma[1, 1] * epsilon[1, 1]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_calculate_K_from_E_and_nu(self) -> None:
        """Test that the bulk modulus is calculated correctly from the Young's modulus and the Poisson's ratio."""
        sut = calculate_K_from_E_and_nu_factory(self._elasticity_state)

        actual = sut(E=self._youngs_modulus, nu=self._poissons_ratio)
        expected = 4 / 5
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_G_from_E_and_nu(self) -> None:
        """Test that the shear modulus is calculated correctly from the Young's modulus and the Poisson's ratio."""

        actual = calculate_G_from_E_and_nu(
            E=self._youngs_modulus, nu=self._poissons_ratio
        )
        expected = 2 / 5
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_E_from_K_and_G(self) -> None:
        """Test that the Young's modulus is calculated correctly from the bulk modulus and the shear modulus."""
        sut = calculate_E_from_K_and_G_factory(self._elasticity_state)

        fake_K = 4 / 5
        fake_G = 2 / 5
        actual = sut(K=fake_K, G=fake_G)
        expected = self._youngs_modulus
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_nu_from_K_and_G(self) -> None:
        """Test that the Poisson's ratio is calculated correctly from the bulk modulus and the shear modulus."""
        sut = calculate_nu_from_K_and_G_factory(self._elasticity_state)

        fake_K = 4 / 5
        fake_G = 2 / 5
        actual = sut(K=fake_K, G=fake_G)
        expected = self._poissons_ratio
        assert_almost_equal(self=self, expected=expected, actual=actual)


class TestMomentumEquation2D_K_G_planeStress(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._elasticity_state = "plane stress"
        self._youngs_modulus = 1.0
        self._poissons_ratio = 0.25
        calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(
            self._elasticity_state
        )
        self._bulk_modulus = calculate_K_from_E_and_nu(
            self._youngs_modulus, self._poissons_ratio
        )
        self._shear_modulus = calculate_G_from_E_and_nu(
            self._youngs_modulus, self._poissons_ratio
        )
        self._factor_volume_force = self._youngs_modulus / (
            (1 - self._poissons_ratio**2)
        )
        PRNG_key = jax.random.PRNGKey(0)
        input_size = 2
        net_ux = set_up_fake_net_ux_for_plane_stress()
        net_uy = set_up_fake_net_uy_for_plane_stress()
        self._bulk_modulus_correction = 0.0
        self._shear_modulus_correction = 0.0
        bulk_modulus_correction_func = set_up_fake_bulk_modulus_correction(self)
        shear_modulus_correction_func = set_up_fake_shear_modulus_correction(self)
        params_net_ux = net_ux.init(PRNG_key, input=jnp.zeros(input_size))
        params_net_uy = net_uy.init(PRNG_key, input=jnp.zeros(input_size))
        params_bulk_modulus_correction = bulk_modulus_correction_func.init(
            PRNG_key, input=jnp.zeros(input_size)
        )
        params_shear_modulus_correction = shear_modulus_correction_func.init(
            PRNG_key, input=jnp.zeros(input_size)
        )
        self._parameters = MomentumEquation2DParameters_K_G(
            net_ux=params_net_ux,
            net_uy=params_net_uy,
            bulk_modulus_correction=params_bulk_modulus_correction,
            shear_modulus_correction=params_shear_modulus_correction,
        )
        self._models = MomentumEquation2DModels_K_G(
            net_ux=net_ux,
            net_uy=net_uy,
            bulk_modulus_correction=bulk_modulus_correction_func,
            shear_modulus_correction=shear_modulus_correction_func,
        )
        self._estimates = MomentumEquation2DEstimates_K_G(
            bulk_modulus=self._bulk_modulus,
            shear_modulus=self._shear_modulus,
        )

    def test_pinn_func_for_positive_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for positive inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([2.0, 2.0])
        training_data = generate_fake_training_data_for_plane_stress(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_pinn_func_for_negative_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for negative inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([-2.0, -2.0])
        training_data = generate_fake_training_data_for_plane_stress(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_pinn_func_for_zero_inputs(self) -> None:
        """
        Test that the pinn function returns zeros for zero inputs.
        """
        sut = pinn_func_factory(self._elasticity_state)
        single_input = jnp.array([0.0, 0.0])
        training_data = generate_fake_training_data_for_plane_stress(self, single_input)

        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            train_data=training_data,
            single_input=single_input,
        )
        expected = jnp.array([0.0, 0.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_traction_func(self) -> None:
        """
        Test that the traction is calculated correctly.
        """
        sut = traction_func_factory(self._elasticity_state)
        single_input = jnp.array([1.0, 2.0])
        normal_vector = jnp.array([0.0, 1.0])
        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            normal_vector=normal_vector,
            single_input=single_input,
        )
        constant_u_x = 8 / 39
        constant_u_y = -44 / 39
        e_xx = (1 / 2) * (4 * constant_u_x + 4 * constant_u_x)
        e_yy = (1 / 2) * (4 * constant_u_y + 4 * constant_u_y)
        e_xy = (1 / 2) * (constant_u_x + 4 * constant_u_y)
        E = self._youngs_modulus
        nu = self._poissons_ratio
        voigt_sigma = (E / (1 - nu**2)) * jnp.array(
            [
                1 * e_xx + nu * e_yy,
                nu * e_xx + 1 * e_yy,
                ((1 - nu) / 2) * 2 * e_xy,
            ]
        )
        sigma = jnp.array(
            [[voigt_sigma[0], voigt_sigma[2]], [voigt_sigma[2], voigt_sigma[1]]]
        )
        expected = jnp.matmul(sigma, normal_vector)
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_strain_energy_func(self) -> None:
        """
        Test that the strain energy is calculated correctly.
        """
        sut = strain_energy_func_factory(self._elasticity_state)
        single_input = jnp.array([1.0, 2.0])
        actual = sut(
            params=self._parameters,
            models=self._models,
            estimates=self._estimates,
            single_input=single_input,
        )
        constant_u_x = 8 / 39
        constant_u_y = -44 / 39
        e_xx = (1 / 2) * (4 * constant_u_x + 4 * constant_u_x)
        e_yy = (1 / 2) * (4 * constant_u_y + 4 * constant_u_y)
        e_xy = (1 / 2) * (constant_u_x + 4 * constant_u_y)
        epsilon = jnp.array([[e_xx, e_xy], [e_xy, e_yy]])
        E = self._youngs_modulus
        nu = self._poissons_ratio
        voigt_sigma = (E / (1 - nu**2)) * jnp.array(
            [
                1 * e_xx + nu * e_yy,
                nu * e_xx + 1 * e_yy,
                ((1 - nu) / 2) * 2 * e_xy,
            ]
        )
        sigma = jnp.array(
            [[voigt_sigma[0], voigt_sigma[2]], [voigt_sigma[2], voigt_sigma[1]]]
        )
        expected = (1 / 2) * (
            sigma[0, 0] * epsilon[0, 0]
            + sigma[0, 1] * epsilon[0, 1]
            + sigma[1, 0] * epsilon[1, 0]
            + sigma[1, 1] * epsilon[1, 1]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_calculate_K_from_E_and_nu(self) -> None:
        """Test that the bulk modulus is calculated correctly from the Young's modulus and the Poisson's ratio."""
        sut = calculate_K_from_E_and_nu_factory(self._elasticity_state)

        actual = sut(E=self._youngs_modulus, nu=self._poissons_ratio)
        expected = 2 / 3
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_G_from_E_and_nu(self) -> None:
        """Test that the shear modulus is calculated correctly from the Young's modulus and the Poisson's ratio."""

        actual = calculate_G_from_E_and_nu(
            E=self._youngs_modulus, nu=self._poissons_ratio
        )
        expected = 2 / 5
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_E_from_K_and_G(self) -> None:
        """Test that the Young's modulus is calculated correctly from the bulk modulus and the shear modulus."""
        sut = calculate_E_from_K_and_G_factory(self._elasticity_state)

        fake_K = 2 / 3
        fake_G = 2 / 5
        actual = sut(K=fake_K, G=fake_G)
        expected = self._youngs_modulus
        assert_almost_equal(self=self, expected=expected, actual=actual)

    def test_calculate_nu_from_K_and_G(self) -> None:
        """Test that the Poisson's ratio is calculated correctly from the bulk modulus and the shear modulus."""
        sut = calculate_nu_from_K_and_G_factory(self._elasticity_state)

        fake_K = 2 / 3
        fake_G = 2 / 5
        actual = sut(K=fake_K, G=fake_G)
        expected = self._poissons_ratio
        assert_almost_equal(self=self, expected=expected, actual=actual)


class TestCaseProtocol(Protocol):
    _bulk_modulus: float
    _shear_modulus: float
    _factor_volume_force: float
    _bulk_modulus_correction: float
    _shear_modulus_correction: float
    _parameters: MomentumEquation2DParameters_K_G
    _models: MomentumEquation2DModels_K_G
    _estimates: MomentumEquation2DEstimates_K_G


def generate_fake_training_data_for_plane_strain(
    test_case: TestCaseProtocol, single_input: JNPArray
) -> TrainingData2D:
    volume_force = jnp.array(
        [
            test_case._shear_modulus * single_input[1],
            2 * test_case._shear_modulus * single_input[0],
        ]
    )
    return TrainingData2D(
        x_data=jnp.array([0.0]),
        y_data_true_ux=jnp.array([0.0]),
        y_data_true_uy=jnp.array([0.0]),
        x_pde=jnp.array([0.0]),
        y_pde_true=jnp.array([0.0]),
        volume_force=volume_force,
        x_traction_bc=jnp.array([0.0]),
        n_traction_bc=jnp.array([0.0]),
        y_traction_bc_true=jnp.array([0.0]),
    )


def generate_fake_training_data_for_plane_stress(
    test_case: TestCaseProtocol, single_input: JNPArray
) -> TrainingData2D:
    volume_force = jnp.array(
        [
            test_case._factor_volume_force * single_input[1],
            2 * test_case._factor_volume_force * single_input[0],
        ]
    )
    return TrainingData2D(
        x_data=jnp.array([0.0]),
        y_data_true_ux=jnp.array([0.0]),
        y_data_true_uy=jnp.array([0.0]),
        x_pde=jnp.array([0.0]),
        y_pde_true=jnp.array([0.0]),
        volume_force=volume_force,
        x_traction_bc=jnp.array([0.0]),
        n_traction_bc=jnp.array([0.0]),
        y_traction_bc_true=jnp.array([0.0]),
    )


def set_up_fake_net_ux_for_plane_strain() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUxPlaneStrain()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_net_uy_for_plane_strain() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUyPlaneStrain()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_net_ux_for_plane_stress() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUxPlaneStress()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_net_uy_for_plane_stress() -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = FakeNetUyPlaneStress()
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_bulk_modulus_correction(
    test_case: TestCaseProtocol,
) -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = ConstantFunction(
            output_size=1,
            func_value_init=hk.initializers.Constant(
                test_case._bulk_modulus_correction
            ),
            name="fake_bulk_modulus_correction",
        )
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))


def set_up_fake_shear_modulus_correction(
    test_case: TestCaseProtocol,
) -> HKTransformed:
    def model_func(input: JNPArray) -> JNPArray:
        model = ConstantFunction(
            output_size=1,
            func_value_init=hk.initializers.Constant(
                test_case._shear_modulus_correction
            ),
            name="fake_shear_modulus_correction",
        )
        return model(input)

    return hk.without_apply_rng(hk.transform(model_func))
