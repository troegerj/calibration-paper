import os
from datetime import date
from time import perf_counter

import numpy as np
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_2D,
)
from parametricpinn.calibration import (
    CalibrationData,
    LeastSquaresConfig,
    calibrate,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data import (
    TrainingDataset2D,
    ValidationDataset2D,
    create_training_dataset_2D,
    create_validation_dataset_2D,
)
from parametricpinn.fem.platewithhole import (
    generate_validation_data as generate_validation_data,
)
from parametricpinn.fem.platewithhole import run_simulation
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfigPWH,
    plot_displacements_pwh,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_2d import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
material_model = "plane stress"
edge_length = 10.0
radius = 2.0
traction_left_x = -150.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
# Network
layer_sizes = [4, 32, 32, 32, 32, 2]
# Training
num_samples_per_parameter = 32
num_collocation_points = 64
number_points_per_bc = 32
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 10000
weight_pde_loss = 1.0
weight_symmetry_bc_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
regenerate_valid_data = True
num_samples_valid = 32
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
fem_mesh_resolution = 0.01
# Calibration
std_high_noise = 4 * 1e-04
std_low_noise = 2 * 1e-04
input_dir_calibration_data = "Paper_Calibration"
input_subdir_high_noise = "with_noise_4e-04"
input_subdir_low_noise = "with_noise_2e-04"
input_subdir_clean = "without_noise"
input_file_high_noise = "displacements_withNoise4e-04.csv"
input_file_low_noise = "displacements_withNoise2e-04.csv"
input_file_clean = "displacements_withoutNoise.csv"
use_least_squares = True
# Input/Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
input_subdirectory_validation = f"{output_date}_validation_data_E_180k_240k_nu_02_04"
output_subdirectory = f"{output_date}_parametric_pinns_linearelasticity"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> tuple[TrainingDataset2D, ValidationDataset2D]:
    def _create_training_dataset() -> TrainingDataset2D:
        print("Generate training data ...")
        traction_left = torch.tensor([traction_left_x, traction_left_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        return create_training_dataset_2D(
            edge_length=edge_length,
            radius=radius,
            traction_left=traction_left,
            volume_force=volume_force,
            min_youngs_modulus=min_youngs_modulus,
            max_youngs_modulus=max_youngs_modulus,
            min_poissons_ratio=min_poissons_ratio,
            max_poissons_ratio=max_poissons_ratio,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            num_samples_per_parameter=num_samples_per_parameter,
        )

    def _create_validation_dataset() -> ValidationDataset2D:
        def _generate_validation_data() -> None:
            def _generate_random_parameter_list(
                size: int, min_value: float, max_value: float
            ) -> list[float]:
                random_params = min_value + torch.rand(size) * (max_value - min_value)
                return random_params.tolist()

            youngs_moduli = _generate_random_parameter_list(
                num_samples_valid, min_youngs_modulus, max_youngs_modulus
            )
            poissons_ratios = _generate_random_parameter_list(
                num_samples_valid, min_poissons_ratio, max_poissons_ratio
            )
            generate_validation_data(
                model=material_model,
                youngs_moduli=youngs_moduli,
                poissons_ratios=poissons_ratios,
                edge_length=edge_length,
                radius=radius,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                traction_left_x=traction_left_x,
                traction_left_y=traction_left_y,
                save_metadata=save_metadata,
                output_subdir=input_subdirectory_validation,
                project_directory=project_directory,
                mesh_resolution=fem_mesh_resolution,
            )

        print("Load validation data ...")
        if regenerate_valid_data:
            print("Run FE simulations to generate validation data ...")
            _generate_validation_data()
        return create_validation_dataset_2D(
            input_subdir=input_subdirectory_validation,
            num_points=num_points_valid,
            num_samples=num_samples_valid,
            project_directory=project_directory,
        )

    training_dataset = _create_training_dataset()
    validation_dataset = _create_validation_dataset()
    return training_dataset, validation_dataset


def create_ansatz() -> StandardAnsatz:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate_x = -edge_length
        max_coordinate_x = 0.0
        min_coordinate_y = 0.0
        max_coordinate_y = edge_length
        min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
        max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

        min_parameters = torch.tensor([min_youngs_modulus, min_poissons_ratio])
        max_parameters = torch.tensor([max_youngs_modulus, max_poissons_ratio])

        min_inputs = torch.concat((min_coordinates, min_parameters))
        max_inputs = torch.concat((max_coordinates, max_parameters))

        _output_subdir = os.path.join(
            output_subdirectory_preprocessing,
            "results_for_determining_normalization_values",
        )
        print("Run FE simulation to determine normalization values ...")
        simulation_results = run_simulation(
            model=material_model,
            youngs_modulus=min_youngs_modulus,
            poissons_ratio=max_poissons_ratio,
            edge_length=edge_length,
            radius=radius,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            traction_left_x=traction_left_x,
            traction_left_y=traction_left_y,
            save_results=False,
            save_metadata=False,
            output_subdir=_output_subdir,
            project_directory=project_directory,
            mesh_resolution=fem_mesh_resolution,
        )
        min_displacement_x = float(np.amin(simulation_results.displacements_x))
        max_displacement_x = float(np.amax(simulation_results.displacements_x))
        min_displacement_y = float(np.amin(simulation_results.displacements_y))
        max_displacement_y = float(np.amax(simulation_results.displacements_y))
        min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
        max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
        print(f"min_inputs: {min_inputs}")
        print(f"max_inputs: {max_inputs}")
        print(f"min_outputs: {min_outputs}")
        print(f"max_outputs: {max_outputs}")
        return {
            "min_inputs": min_inputs.to(device),
            "max_inputs": max_inputs.to(device),
            "min_outputs": min_outputs.to(device),
            "max_outputs": max_outputs.to(device),
        }

    normalization_values = _determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_2D(
        displacement_x_right=torch.tensor(0.0).to(device),
        displacement_y_bottom=torch.tensor(0.0).to(device),
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_outputs"],
        max_outputs=normalization_values["max_outputs"],
        network=network,
    ).to(device)


ansatz = create_ansatz()


def training_step() -> None:
    train_config = TrainingConfiguration(
        ansatz=ansatz,
        material_model=material_model,
        number_points_per_bc=number_points_per_bc,
        weight_pde_loss=weight_pde_loss,
        weight_symmetry_bc_loss=weight_symmetry_bc_loss,
        weight_traction_bc_loss=weight_traction_bc_loss,
        training_dataset=training_dataset,
        number_training_epochs=number_training_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        validation_interval=validation_interval,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacement_fields() -> None:
        displacements_plotter_config = DisplacementsPlotterConfigPWH()

        plot_displacements_pwh(
            ansatz=ansatz,
            youngs_modulus_and_poissons_ratio_list=[
                (210000, 0.3),
                (180000, 0.2),
                (180000, 0.4),
                (240000, 0.2),
                (240000, 0.4),
            ],
            model=material_model,
            edge_length=edge_length,
            radius=radius,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            traction_left_x=traction_left_x,
            traction_left_y=traction_left_y,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
            mesh_resolution=0.01,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step(input_subdir: str, input_file_name: str, std_noise: float) -> None:
    print("Start calibration ...")

    def load_calibration_data() -> tuple[Tensor, Tensor]:
        input_subdir_path = os.path.join(input_dir_calibration_data, input_subdir)
        data_reader = CSVDataReader(project_directory)
        data = torch.from_numpy(data_reader.read(input_file_name, input_subdir_path))
        coordinates = data[:, :2].to(device)
        noisy_displacements = data[:, 2:].to(device)

        return coordinates, noisy_displacements

    def modifiy_coordinates(coordinates: Tensor) -> Tensor:
        num_data_points = coordinates.size()[0]
        return coordinates - torch.tensor(
            [edge_length, 0.0], dtype=torch.float64, device=device
        ).repeat((num_data_points, 1))

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    coordinates, noisy_displacements = load_calibration_data()
    coordinates = modifiy_coordinates(coordinates)
    data = CalibrationData(
        inputs=coordinates,
        outputs=noisy_displacements,
        std_noise=std_noise,
    )

    initial_parameters = torch.tensor(
        [min_youngs_modulus, min_poissons_ratio], device=device
    )

    least_squares_config = LeastSquaresConfig(
        initial_parameters=initial_parameters,
        num_iterations=int(1e2),
        ansatz=model,
        calibration_data=data,
    )

    if use_least_squares:
        start = perf_counter()
        identified_parameters_ls, loss_hist_ls = calibrate(
            calibration_config=least_squares_config,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Identified parameter: {identified_parameters_ls}")
        print(f"Run time least squares: {time}")
    print("Calibration finished.")


if retrain_parametric_pinn:
    training_dataset, validation_dataset = create_datasets()
    training_step()
print("Clean data:")
calibration_step(
    input_subdir=input_subdir_clean,
    input_file_name=input_file_clean,
    std_noise=1e-8,  # Dummy value because 0.0 causes error,
)
print("Low noise:")
calibration_step(
    input_subdir=input_subdir_low_noise,
    input_file_name=input_file_low_noise,
    std_noise=std_low_noise,
)
print("High noise:")
calibration_step(
    input_subdir=input_subdir_high_noise,
    input_file_name=input_file_high_noise,
    std_noise=std_high_noise,
)
