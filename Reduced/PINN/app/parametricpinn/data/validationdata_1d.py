from typing import cast

import torch
from torch.utils.data import Dataset

from parametricpinn.data.dataset import Dataset
from parametricpinn.data.geometry import StretchedRod
from parametricpinn.types import Tensor


def calculate_displacements_solution_1D(
    coordinates: Tensor | float,
    length: float,
    youngs_modulus: Tensor | float,
    traction: float,
    volume_force: float,
) -> Tensor | float:
    return (traction / youngs_modulus) * coordinates + (
        volume_force / youngs_modulus
    ) * (length * coordinates - 1 / 2 * coordinates**2)


class ValidationDataset1D(Dataset):
    def __init__(
        self,
        geometry: StretchedRod,
        traction: float,
        volume_force: float,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        num_points: int,
        num_samples: int,
    ) -> None:
        self._geometry = geometry
        self._traction = traction
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points = num_points
        self._num_samples = num_samples
        self._samples_x: list[Tensor] = []
        self._samples_y_true: list[Tensor] = []

        self._generate_samples()

    def _generate_samples(self) -> None:
        for i in range(self._num_samples):
            youngs_modulus = self._generate_random_youngs_modulus()
            coordinates = self._generate_random_coordinates()
            self._add_input_sample(coordinates, youngs_modulus)
            self._add_output_sample(coordinates, youngs_modulus)

    def _generate_random_youngs_modulus(self) -> Tensor:
        return self._min_youngs_modulus + torch.rand((1)) * (
            self._max_youngs_modulus - self._min_youngs_modulus
        )

    def _generate_random_coordinates(self) -> Tensor:
        return self._geometry.create_random_points(self._num_points)

    def _add_input_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
        x_coor = coordinates
        x_E = self._repeat_tensor(youngs_modulus, (self._num_points, 1))
        x = torch.concat((x_coor, x_E), dim=1)
        self._samples_x.append(x)

    def _add_output_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
        y_true = calculate_displacements_solution_1D(
            coordinates=coordinates,
            length=self._geometry.length,
            youngs_modulus=youngs_modulus,
            traction=self._traction,
            volume_force=self._volume_force,
        )
        self._samples_y_true.append(cast(Tensor, y_true))

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true


def collate_validation_data_1D(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor]:
    x_batch = []
    y_true_batch = []

    for sample_x, sample_y_true in batch:
        x_batch.append(sample_x)
        y_true_batch.append(sample_y_true)

    batch_x = torch.concat(x_batch, dim=0)
    batch_y_true = torch.concat(y_true_batch, dim=0)
    return batch_x, batch_y_true


def create_validation_dataset_1D(
    length: float,
    traction: float,
    volume_force: float,
    min_youngs_modulus: float,
    max_youngs_modulus: float,
    num_points: int,
    num_samples: int,
):
    geometry = StretchedRod(length=length)
    return ValidationDataset1D(
        geometry=geometry,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points=num_points,
        num_samples=num_samples,
    )
