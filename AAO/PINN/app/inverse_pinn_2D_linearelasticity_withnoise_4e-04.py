# Standard library imports
from datetime import date
from functools import partial
import os
from time import perf_counter
from typing import Any, TypeAlias

# Third-party imports
import haiku as hk
import jax
import jax.numpy as jnp
import optax

# Local library imports
from calibrationpinn import Settings
from calibrationpinn.domains import (
    DomainBuilder2D,
    InputDataReader2D,
    TrainingData2D,
    ValidationData2D,
)
from calibrationpinn.domains.dataselection import select_data_randomly
from calibrationpinn.inputoutput import (
    CSVDataReader,
    PandasDataWriter,
    PathAdministrator,
)
from calibrationpinn.inputoutput.preprocessing import (
    preprocess_calibration_paper_data_2D,
)
from calibrationpinn.models import ModelBuilder
from calibrationpinn.simulations import (
    Domain2DWithHolePlotter,
    Domain2DWithHolePlotterConfig,
    Simulator2D,
)
from calibrationpinn.training.errormetrics import (
    mean_squared_error,
    relative_l2_norm,
    relative_mean_squared_error,
)
from calibrationpinn.training.loggers import (
    should_model_be_validated,
    TrainingLogger,
    TrainingLoggerPlotter,
    TrainingLoggerPlotterConfig,
)
from calibrationpinn.training.optimizers import BFGS
from calibrationpinn.training.pinns.momentumEquation2D_K_G import (
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_E_from_K_and_G_factory,
    calculate_nu_from_K_and_G_factory,
    material_parameter_func,
    pinn_func_factory,
    strain_energy_func_factory,
    traction_func_factory,
    MomentumEquation2DEstimates_K_G,
    MomentumEquation2DModels_K_G,
    MomentumEquation2DParameters_K_G,
)
from calibrationpinn.typeAliases import (
    HKTransformed,
    JNPArray,
    JNPFloat,
    JNPPyTree,
    NPFloat,
    PRNGKey,
)


Parameters: TypeAlias = MomentumEquation2DParameters_K_G
Models: TypeAlias = MomentumEquation2DModels_K_G
Estimates: TypeAlias = MomentumEquation2DEstimates_K_G
TrainingData: TypeAlias = TrainingData2D
ValidationData: TypeAlias = ValidationData2D
Net: TypeAlias = HKTransformed


noise_level = "with_noise_4e-04"


def run_2D_FEM_plate_with_hole_simulation(
    youngs_modulus_estimate: float,
    poissons_ratio_estimate: float,
    num_data_points_train: int,
    num_collocation_points_train: int,
    output_subdir_name: str,
    PRNG_key: PRNGKey,
):
    ################################################################################
    ##### Input data preprocessing
    input_file_name = "displacements_withNoise4e-04.csv"
    input_subdir_name = os.path.join("Paper_Calibration", noise_level)
    preprocess_calibration_paper_data_2D(input_file_name, input_subdir_name)

    ################################################################################
    ##### Configuration
    elasticity_state = "plane stress"
    calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(elasticity_state)
    bulk_modulus_est = calculate_K_from_E_and_nu(
        youngs_modulus_estimate, poissons_ratio_estimate
    )
    shear_modulus_est = calculate_G_from_E_and_nu(
        youngs_modulus_estimate, poissons_ratio_estimate
    )
    calculate_E_from_K_and_G = calculate_E_from_K_and_G_factory(elasticity_state)
    calculate_nu_from_K_and_G = calculate_nu_from_K_and_G_factory(elasticity_state)
    bulk_modulus_cor = 0.0
    shear_modulus_cor = 0.0
    ### Domain
    side_length_plate = 10.0
    radius_hole = 2.0
    area_plate = side_length_plate**2 - ((1 / 4) * jnp.pi * radius_hole**2)
    ### Training
    max_epochs = 100000
    valid_interval = 1
    num_nonhomogeneous_traction_bc_points = 64
    num_points_valid = 3097
    num_points_per_edge_sim = 256
    input_size = 2
    ### Loss function
    weight_data_loss = 1e3

    ################################################################################
    ##### Setup
    settings = Settings()
    path_admin = PathAdministrator(settings)
    pandas_data_writer = PandasDataWriter(path_admin)
    PRNG_keys = jax.random.split(PRNG_key, num=4)
    PRNG_key_net_ux = PRNG_keys[0]
    PRNG_key_net_uy = PRNG_keys[1]
    PRNG_key_bulk_modulus_correction_func = PRNG_keys[2]
    PRNG_key_shear_modulus_correction_func = PRNG_keys[3]

    ### Domain
    data_reader = CSVDataReader(path_admin)
    input_data_reader = InputDataReader2D(data_reader, input_subdir_name)
    domain_builder = DomainBuilder2D()
    domain = domain_builder.build_domain_without_solution(
        input_reader=input_data_reader,
        proportion_training_data=1.0,
        data_selection_func=select_data_randomly,
        PRNG_key=PRNG_key,
    )

    ### Data
    train_data = domain.generate_training_data(
        num_data_points_train, num_collocation_points_train
    )
    valid_data = domain.generate_validation_data(num_points_valid)
    sim_data = domain.generate_simulation_data(num_points_per_edge_sim)

    ### Models
    model_builder = ModelBuilder()

    # Neural networks for displacement field approximation:
    # Input: coordinate x, coordinate y
    # Output: displacement x
    # hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal') = glorot normal
    net_ux = model_builder.build_normalized_feedforward_neural_network(
        output_sizes=[16, 16, 1],
        w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        b_init=hk.initializers.Constant(0.0),
        activation=jnp.tanh,
        reference_inputs=train_data.x_data,
        reference_outputs=train_data.y_data_true_ux,
        name="net_ux",
    )
    params_net_ux = net_ux.init(PRNG_key_net_ux, input=jnp.zeros(input_size))

    # Output: displacement y
    net_uy = model_builder.build_normalized_feedforward_neural_network(
        output_sizes=[16, 16, 1],
        w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        b_init=hk.initializers.Constant(0.0),
        activation=jnp.tanh,
        reference_inputs=train_data.x_data,
        reference_outputs=train_data.y_data_true_uy,
        name="net_uy",
    )
    params_net_uy = net_uy.init(PRNG_key_net_uy, input=jnp.zeros(input_size))

    # Constant function for correction of bulk modulus estimate:
    # Input: coordinate x, coordinate y
    bulk_modulus_correction_func = model_builder.build_constant_function(
        output_size=1,
        func_value_init=hk.initializers.Constant(bulk_modulus_cor),
        name="bulk_modulus_correction_func",
    )
    params_bulk_modulus_correction_func = bulk_modulus_correction_func.init(
        PRNG_key_bulk_modulus_correction_func, input=jnp.zeros(input_size)
    )

    # Constant function for correction of shear modulus estimate:
    shear_modulus_correction_func = model_builder.build_constant_function(
        output_size=1,
        func_value_init=hk.initializers.Constant(shear_modulus_cor),
        name="shear_modulus_correction_func",
    )
    params_shear_modulus_correction_func = shear_modulus_correction_func.init(
        PRNG_key_shear_modulus_correction_func, input=jnp.zeros(input_size)
    )

    ### Optimizer
    optimizer = BFGS(zoom_maxiter=50)

    ### Loss metric
    mean_displacement_x = jnp.mean(jnp.absolute(train_data.y_data_true_ux))
    mean_displacement_y = jnp.mean(jnp.absolute(train_data.y_data_true_uy))
    print(f"Mean displacement x: {mean_displacement_x}")
    print(f"Mean displacement y: {mean_displacement_y}")
    loss_metric_data_ux = relative_mean_squared_error(mean_displacement_x)
    loss_metric_data_uy = relative_mean_squared_error(mean_displacement_y)

    ### Loss functions
    pinn_func = pinn_func_factory(elasticity_state)
    strain_energy_func = strain_energy_func_factory(elasticity_state)
    traction_func = traction_func_factory(elasticity_state)

    def losses_data_func(
        params: Parameters, models: Models, train_data: TrainingData
    ) -> tuple[JNPArray, ...]:
        def loss_data_func_one_net(params_net, net, x_data, y_data_true, loss_metric):
            vmap_apply_net = lambda single_input: net.apply(params_net, single_input)
            y_data_pred = jax.vmap(vmap_apply_net)(x_data)
            return loss_metric(y_data_true, y_data_pred)

        loss_data_ux = weight_data_loss * loss_data_func_one_net(
            params.net_ux,
            models.net_ux,
            train_data.x_data,
            train_data.y_data_true_ux,
            loss_metric_data_ux,
        )
        loss_data_uy = weight_data_loss * loss_data_func_one_net(
            params.net_uy,
            models.net_uy,
            train_data.x_data,
            train_data.y_data_true_uy,
            loss_metric_data_uy,
        )
        return loss_data_ux, loss_data_uy

    def loss_pde_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        train_data: TrainingData,
    ) -> JNPArray:
        vmap_apply_pinn_func = lambda single_input: pinn_func(
            params, models, estimates, train_data, single_input
        )
        y_pde_pred = jax.vmap(vmap_apply_pinn_func)(train_data.x_pde)
        return mean_squared_error(train_data.y_pde_true, y_pde_pred)

    # def loss_traction_bc_func(
    #     params: Parameters,
    #     models: Models,
    #     estimates: Estimates,
    #     train_data: TrainingData,
    # ) -> JNPArray:
    #     vmap_apply_traction_func = lambda single_normal, single_input: traction_func(
    #         params, models, estimates, single_normal, single_input
    #     )
    #     y_traction_bc_pred = jax.vmap(vmap_apply_traction_func)(
    #         train_data.n_traction_bc, train_data.x_traction_bc
    #     )
    #     return mean_squared_error(train_data.y_traction_bc_true, y_traction_bc_pred)

    def loss_energy_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        train_data: TrainingData,
    ) -> JNPArray:
        def calculate_internal_energy() -> JNPFloat:
            vmap_apply_strain_energy_func = lambda single_input: strain_energy_func(
                params, models, estimates, single_input
            )
            summed_strain_energy = jnp.sum(
                jax.vmap(vmap_apply_strain_energy_func)(train_data.x_pde)
            )
            return 2 * (area_plate / train_data.x_pde.shape[0]) * summed_strain_energy

        def calculate_external_energy() -> JNPFloat:
            x_traction_bc = train_data.x_traction_bc[
                :num_nonhomogeneous_traction_bc_points, :
            ]
            y_traction_bc_true = train_data.y_traction_bc_true[
                :num_nonhomogeneous_traction_bc_points, :
            ]

            def calculate_displacements(
                params_net: JNPPyTree, net: Net, x_data: JNPArray
            ) -> JNPArray:
                vmap_apply_net = lambda single_input: net.apply(
                    params_net, single_input
                )
                return jax.vmap(vmap_apply_net)(x_data)

            displacements_x = calculate_displacements(
                params.net_ux, models.net_ux, x_traction_bc
            )
            displacements_y = calculate_displacements(
                params.net_uy, models.net_uy, x_traction_bc
            )
            displacements = jnp.hstack((displacements_x, displacements_y))
            summed_external_energy = jnp.sum(
                jnp.einsum("ij,ij->i", y_traction_bc_true, displacements)
            )
            return (side_length_plate / x_traction_bc.shape[0]) * summed_external_energy

        return mean_squared_error(
            jnp.array([0.0]),
            jnp.array([calculate_internal_energy() - calculate_external_energy()]),
        )

    @partial(jax.jit, static_argnums=(1,))
    def loss_func(
        params: Parameters,
        models: Models,
        estimates: Estimates,
        train_data: TrainingData,
    ) -> JNPArray:
        loss_data_ux, loss_data_uy = losses_data_func(params, models, train_data)
        loss_pde = loss_pde_func(params, models, estimates, train_data)
        loss_energy = loss_energy_func(params, models, estimates, train_data)
        # loss_traction_bc = loss_traction_bc_func(params, models, estimates, train_data)
        return loss_data_ux + loss_data_uy + loss_pde + loss_energy

    ### Training
    @partial(jax.jit, static_argnums=(2,))
    def train_step(
        opt_state: Any,
        params: Parameters,
        models: Models,
        estimates: Estimates,
        train_data: TrainingData,
    ) -> tuple[tuple[JNPArray, ...], Any, Parameters]:
        loss_data_ux, loss_data_uy = losses_data_func(params, models, train_data)
        loss_pde = loss_pde_func(params, models, estimates, train_data)
        loss_energy = loss_energy_func(params, models, estimates, train_data)
        # loss_traction_bc = loss_traction_bc_func(params, models, estimates, train_data)
        losses = (loss_data_ux, loss_data_uy, loss_pde, loss_energy)
        updates, opt_state = optimizer.update(
            params=params,
            loss_func=loss_func,
            func_args=(models, estimates, train_data),
            optimizer_state=opt_state,
        )
        params = optax.apply_updates(params, updates)
        return losses, opt_state, params

    ### Validation
    @partial(jax.jit, static_argnums=(1,))
    def valid_step(
        params: Parameters, models: Models, valid_data: ValidationData
    ) -> tuple[JNPArray, ...]:
        def error_func(
            params: Parameters, models: Models, valid_data: ValidationData
        ) -> tuple[JNPArray, ...]:
            rL2_ux = relative_L2_norm_func(
                params.net_ux,
                models.net_ux,
                valid_data.x_data,
                valid_data.y_data_true_ux,
            )
            rL2_uy = relative_L2_norm_func(
                params.net_uy,
                models.net_uy,
                valid_data.x_data,
                valid_data.y_data_true_uy,
            )
            return (rL2_ux, rL2_uy)

        def relative_L2_norm_func(
            params_net: JNPPyTree, net: Net, x_data: JNPArray, y_data_true: JNPArray
        ) -> JNPArray:
            vmap_apply_net = lambda single_input: net.apply(params_net, single_input)
            y_data_pred = jax.vmap(vmap_apply_net)(x_data)
            return relative_l2_norm(y_data_true, y_data_pred)

        return error_func(params, models, valid_data)

    def calculate_E_and_nu(
        bulk_modulus_cor: NPFloat, shear_modulus_cor: NPFloat
    ) -> tuple[float, ...]:
        identified_K = material_parameter_func(
            float(bulk_modulus_cor), bulk_modulus_est
        )
        identified_G = material_parameter_func(
            float(shear_modulus_cor), shear_modulus_est
        )
        identified_E = calculate_E_from_K_and_G(identified_K, identified_G)
        identified_nu = calculate_nu_from_K_and_G(identified_K, identified_G)
        return identified_E, identified_nu

    ################################################################################
    ##### Main
    params = MomentumEquation2DParameters_K_G(
        net_ux=params_net_ux,
        net_uy=params_net_uy,
        bulk_modulus_correction=params_bulk_modulus_correction_func,
        shear_modulus_correction=params_shear_modulus_correction_func,
    )

    models = MomentumEquation2DModels_K_G(
        net_ux=net_ux,
        net_uy=net_uy,
        bulk_modulus_correction=bulk_modulus_correction_func,
        shear_modulus_correction=shear_modulus_correction_func,
    )

    estimates = MomentumEquation2DEstimates_K_G(
        bulk_modulus=bulk_modulus_est, shear_modulus=shear_modulus_est
    )

    opt_state = optimizer.init(
        params=params, loss_func=loss_func, func_args=(models, estimates, train_data)
    )

    logger = TrainingLogger(
        loss_names=(
            "loss_ux",
            "loss_uy",
            "loss_pde",
            "loss_energy",
        ),
        error_metric_names=("rL2_ux", "rL2_uy"),
        parameter_names=("K_cor", "G_cor", "E", "nu"),
        pandas_data_writer=pandas_data_writer,
        settings=settings,
        additional_log_names=("is_terminated", "status"),
    )

    ### Train PINN
    print("Training started ...")
    for epoch in range(1, max_epochs + 1):
        if opt_state.is_terminated:
            break
        start_time = perf_counter()
        losses, opt_state, params = train_step(
            opt_state, params, models, estimates, train_data
        )
        end_time = perf_counter()
        training_time = end_time - start_time
        if should_model_be_validated(epoch, max_epochs, valid_interval):
            errors = valid_step(params, models, valid_data)
            bulk_modulus_correction = params.bulk_modulus_correction[
                "bulk_modulus_correction_func"
            ]["func_value"][0]
            shear_modulus_correction = params.shear_modulus_correction[
                "shear_modulus_correction_func"
            ]["func_value"][0]
            identified_E, identified_nu = calculate_E_and_nu(
                bulk_modulus_correction, shear_modulus_correction
            )
            parameters = (
                bulk_modulus_correction,
                shear_modulus_correction,
                identified_E,
                identified_nu,
            )
            additional_logs = (str(opt_state.is_terminated), str(opt_state.status))
            logger.log(
                epoch, losses, errors, parameters, training_time, additional_logs
            )

    logger.save_as_csv("training_logger", output_subdir_name)

    ### Postprocessing
    print("Postprocessing ...")
    # Plot training logger
    plotter_config_losses = TrainingLoggerPlotterConfig(
        title="Losses",
        save_title="losses",
        x_label="epochs [-]",
        y_label="(rel.) mean squared error",
    )
    plotter_config_errors = TrainingLoggerPlotterConfig(
        title="Errors",
        save_title="errors",
        x_label="epochs [-]",
        y_label="rel. " + r"$L^{2}$" + " norm",
    )
    plotter_config_params = TrainingLoggerPlotterConfig(
        title="Identified correction factors",
        save_title="material_parameters",
        x_label="epochs [-]",
        y_label="material parameter",
    )

    plotter = TrainingLoggerPlotter(output_subdir_name, path_admin)
    plotter.plot(
        logger, plotter_config_losses, plotter_config_errors, plotter_config_params
    )

    # Simulations
    plotter_simulation = Domain2DWithHolePlotter(output_subdir_name, path_admin)
    simulator = Simulator2D(plotter_simulation)

    plotter_config_displacements_x = Domain2DWithHolePlotterConfig(
        simulation_object="displacements x",
        save_title_identifier="displacements_x",
        radius_hole=radius_hole,
    )
    plotter_config_displacements_y = Domain2DWithHolePlotterConfig(
        simulation_object="displacements y",
        save_title_identifier="displacements_y",
        radius_hole=radius_hole,
    )

    simulator.simulate_displacements(
        params,
        models,
        sim_data,
        plotter_config_displacements_x,
        plotter_config_displacements_y,
    )

    return logger


if __name__ == "__main__":
    youngs_modulus_estimate = 210000
    poissons_ratio_estimate = 0.3
    num_data_points_train = 3097
    num_collocation_points_train = num_data_points_train
    current_date = date.today()
    formatted_current_date = current_date.strftime("%Y%m%d")
    output_subdir_name = os.path.join(
        "Paper_Calibration",
        noise_level,
        f"{formatted_current_date}_Simulation",
    )
    PRNG_key = jax.random.PRNGKey(0)

    run_2D_FEM_plate_with_hole_simulation(
        youngs_modulus_estimate,
        poissons_ratio_estimate,
        num_data_points_train,
        num_collocation_points_train,
        output_subdir_name,
        PRNG_key,
    )
