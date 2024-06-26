import statistics
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.data import (
    TrainingData2DCollocation,
    TrainingData2DSymmetryBC,
    TrainingData2DTractionBC,
    TrainingDataset2D,
    ValidationDataset2D,
    collate_training_data_2D,
    collate_validation_data_2D,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelSaver
from parametricpinn.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_loss_history,
    plot_valid_history,
)
from parametricpinn.training.loss_2d import (
    momentum_equation_func_factory,
    stress_func_factory,
    traction_func_factory,
)
from parametricpinn.training.metrics import mean_absolute_error, relative_l2_norm
from parametricpinn.types import Device, Tensor


@dataclass
class TrainingConfiguration:
    ansatz: StandardAnsatz
    material_model: str
    number_points_per_bc: int
    weight_pde_loss: float
    weight_symmetry_bc_loss: float
    weight_traction_bc_loss: float
    training_dataset: TrainingDataset2D
    number_training_epochs: int
    training_batch_size: int
    validation_dataset: ValidationDataset2D
    validation_interval: int
    output_subdirectory: str
    project_directory: ProjectDirectory
    device: Device


def train_parametric_pinn(train_config: TrainingConfiguration) -> None:
    ansatz = train_config.ansatz
    material_model = train_config.material_model
    num_points_per_bc = train_config.number_points_per_bc
    weight_pde_loss = train_config.weight_pde_loss
    weight_symmetry_bc_loss = train_config.weight_symmetry_bc_loss
    weight_traction_bc_loss = train_config.weight_traction_bc_loss
    train_dataset = train_config.training_dataset
    train_num_epochs = train_config.number_training_epochs
    train_batch_size = train_config.training_batch_size
    valid_dataset = train_config.validation_dataset
    valid_batch_size = len(valid_dataset)
    valid_interval = train_config.validation_interval
    output_subdir = train_config.output_subdirectory
    project_directory = train_config.project_directory
    device = train_config.device

    loss_metric = torch.nn.MSELoss()

    ### Loss function
    momentum_equation_func = momentum_equation_func_factory(material_model)
    stress_func = stress_func_factory(material_model)
    traction_func = traction_func_factory(material_model)
    # strain_energy_func = strain_energy_func_factory(material_model)
    # traction_energy_func = traction_energy_func_factory(material_model)

    lambda_pde_loss = torch.tensor(weight_pde_loss, requires_grad=True).to(device)
    lambda_symmetry_bc_loss = torch.tensor(
        weight_symmetry_bc_loss, requires_grad=True
    ).to(device)
    lambda_traction_bc_loss = torch.tensor(
        weight_traction_bc_loss, requires_grad=True
    ).to(device)
    # lambda_energy_loss = torch.tensor(weight_energy_loss, requires_grad=True).to(device)

    def loss_func(
        ansatz: StandardAnsatz,
        collocation_data: TrainingData2DCollocation,
        symmetry_bc_data: TrainingData2DSymmetryBC,
        traction_bc_data: TrainingData2DTractionBC,
    ) -> tuple[Tensor, Tensor, Tensor]:
        def loss_func_pde(
            ansatz: StandardAnsatz, collocation_data: TrainingData2DCollocation
        ) -> Tensor:
            x_coor = collocation_data.x_coor.to(device)
            x_E = collocation_data.x_E
            x_nu = collocation_data.x_nu
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            volume_force = collocation_data.f.to(device)
            y_true = torch.zeros_like(x_coor).to(device)
            y = momentum_equation_func(ansatz, x_coor, x_param, volume_force)
            return loss_metric(y_true, y)

        def loss_func_symmetry_bc(
            ansatz: StandardAnsatz, symmetry_bc_data: TrainingData2DSymmetryBC
        ) -> Tensor:
            x_coor = symmetry_bc_data.x_coor.to(device)
            x_E = symmetry_bc_data.x_E
            x_nu = symmetry_bc_data.x_nu
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            shear_stress_filter = (
                torch.tensor([[0.0, 1.0], [1.0, 0.0]])
                .repeat(train_batch_size * 2 * num_points_per_bc, 1, 1)
                .to(device)
            )
            stress_tensors = stress_func(ansatz, x_coor, x_param)
            y = shear_stress_filter * stress_tensors
            y_true = (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]])
                .repeat(train_batch_size * 2 * num_points_per_bc, 1, 1)
                .to(device)
            )
            return loss_metric(y_true, y)

        def loss_func_traction_bc(
            ansatz: StandardAnsatz, traction_bc_data: TrainingData2DTractionBC
        ) -> Tensor:
            x_coor = traction_bc_data.x_coor.to(device)
            x_E = traction_bc_data.x_E
            x_nu = traction_bc_data.x_nu
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            normal = traction_bc_data.normal.to(device)
            y_true = traction_bc_data.y_true.to(device)
            y = traction_func(ansatz, x_coor, x_param, normal)
            return loss_metric(y_true, y)

        # def loss_func_energy(
        #     ansatz: StandardAnsatz,
        #     collocation_data: TrainingData2DCollocation,
        #     traction_bc_data: TrainingData2DTractionBC,
        # ) -> Tensor:
        #     x_coor_int = collocation_data.x_coor.to(device)
        #     x_E_int = collocation_data.x_E
        #     x_nu_int = collocation_data.x_nu
        #     x_param_int = torch.concat((x_E_int, x_nu_int), dim=1).to(device)
        #     strain_energy = strain_energy_func(ansatz, x_coor_int, x_param_int, area_pwh)

        #     x_coor_ext = traction_bc_data.x_coor.to(device)
        #     x_E_ext = traction_bc_data.x_E
        #     x_nu_ext = traction_bc_data.x_nu
        #     x_param_ext = torch.concat((x_E_ext, x_nu_ext), dim=1).to(device)
        #     normal_ext = traction_bc_data.normal.to(device)
        #     area_frac_ext = traction_bc_data.area_frac.to(device)
        #     traction_energy = traction_energy_func(
        #         ansatz, x_coor_ext, x_param_ext, normal_ext, area_frac_ext
        #     )
        #     y = strain_energy - traction_energy
        #     y_true = torch.tensor(0.0).to(device)
        #     return loss_metric(y_true, y)

        loss_pde = lambda_pde_loss * loss_func_pde(ansatz, collocation_data)
        loss_symmetry_bc = lambda_symmetry_bc_loss * loss_func_symmetry_bc(
            ansatz, symmetry_bc_data
        )
        loss_traction_bc = lambda_traction_bc_loss * loss_func_traction_bc(
            ansatz, traction_bc_data
        )
        # loss_energy = lambda_energy_loss * loss_func_energy(
        #     ansatz, collocation_data, traction_bc_data
        # )
        return loss_pde, loss_symmetry_bc, loss_traction_bc  # , loss_energy

    ### Validation
    def validate_model(
        ansatz: StandardAnsatz, valid_dataloader: DataLoader
    ) -> tuple[float, float]:
        ansatz.eval()
        with torch.no_grad():
            valid_batches = iter(valid_dataloader)
            mae_hist_batches = []
            rl2_hist_batches = []

            for x, y_true in valid_batches:
                x = x.to(device)
                y_true = y_true.to(device)
                y = ansatz(x)
                mae_batch = mean_absolute_error(y_true, y)
                rl2_batch = relative_l2_norm(y_true, y)
                mae_hist_batches.append(mae_batch.cpu().item())
                rl2_hist_batches.append(rl2_batch.cpu().item())

            mean_mae = statistics.mean(mae_hist_batches)
            mean_rl2 = statistics.mean(rl2_hist_batches)
        return mean_mae, mean_rl2

    ### Training process
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_training_data_2D,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_validation_data_2D,
    )

    optimizer = torch.optim.LBFGS(
        params=ansatz.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_hist_pde = []
    loss_hist_symmetry_bc = []
    loss_hist_traction_bc = []
    # loss_hist_energy = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        # loss_collocation, loss_symmetry_bc, loss_traction_bc, loss_energy = loss_func(
        #     ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
        # )
        loss_collocation, loss_symmetry_bc, loss_traction_bc = loss_func(
            ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
        )
        loss = loss_collocation + loss_symmetry_bc + loss_traction_bc  # + loss_energy
        loss.backward(retain_graph=True)
        return loss.item()

    print("Start training ...")
    for epoch in range(train_num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_symmetry_bc_batches = []
        loss_hist_traction_bc_batches = []
        # loss_hist_energy_batches = []

        for batch_collocation, batch_symmetry_bc, batch_traction_bc in train_batches:
            ansatz.train()

            # Forward pass
            # loss_pde, loss_symmetry_bc, loss_traction_bc, loss_energy = loss_func(
            #     ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
            # )
            loss_pde, loss_symmetry_bc, loss_traction_bc = loss_func(
                ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
            )

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_symmetry_bc_batches.append(loss_symmetry_bc.detach().cpu().item())
            loss_hist_traction_bc_batches.append(loss_traction_bc.detach().cpu().item())
            # loss_hist_energy_batches.append(loss_energy.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_symmetry_bc = statistics.mean(loss_hist_symmetry_bc_batches)
        mean_loss_traction_bc = statistics.mean(loss_hist_traction_bc_batches)
        # mean_loss_energy = statistics.mean(loss_hist_energy_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_symmetry_bc.append(mean_loss_symmetry_bc)
        loss_hist_traction_bc.append(mean_loss_traction_bc)
        # loss_hist_energy.append(mean_loss_energy)

        print("##################################################")
        print(f"Epoch {epoch} / {train_num_epochs - 1}")
        print(f"PDE: \t\t {mean_loss_pde}")
        print(f"SYMMETRY_BC: \t {mean_loss_symmetry_bc}")
        print(f"TRACTION_BC: \t {mean_loss_traction_bc}")
        # print(f"ENERGY: \t {mean_loss_energy}")
        print("##################################################")
        if epoch % valid_interval == 0 or epoch == train_num_epochs:
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)
            print(
                f"Validation: Epoch {epoch} / {train_num_epochs - 1}, MAE: {mae}, rL2: {rl2}"
            )

    ### Postprocessing
    print("Postprocessing ...")
    history_plotter_config = HistoryPlotterConfig()

    # plot_loss_history(
    #     loss_hists=[
    #         loss_hist_pde,
    #         loss_hist_symmetry_bc,
    #         loss_hist_traction_bc,
    #         loss_hist_energy,
    #     ],
    #     loss_hist_names=["PDE", "Symmetry BC", "Traction BC", "ENERGY"],
    #     file_name="loss_pinn.png",
    #     output_subdir=output_subdir,
    #     project_directory=project_directory,
    #     config=history_plotter_config,
    # )
    plot_loss_history(
        loss_hists=[
            loss_hist_pde,
            loss_hist_symmetry_bc,
            loss_hist_traction_bc,
        ],
        loss_hist_names=["PDE", "Symmetry BC", "Traction BC"],
        file_name="loss_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    plot_valid_history(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mae,
        valid_metric="mean absolute error",
        file_name="mae_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    plot_valid_history(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_rl2,
        valid_metric="rel. L2 norm",
        file_name="rl2_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    ### Save model
    print("Save model ...")
    model_saver = PytorchModelSaver(project_directory=project_directory)
    model_saver.save(
        model=ansatz,
        file_name="model_parameters",
        subdir_name=output_subdir,
        device=device,
    )
