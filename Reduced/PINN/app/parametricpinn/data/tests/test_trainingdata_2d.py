import math

import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon, box

from parametricpinn.data import (
    TrainingData2DCollocation,
    TrainingData2DSymmetryBC,
    TrainingData2DTractionBC,
    TrainingDataset2D,
    collate_training_data_2D,
)
from parametricpinn.data.geometry import PlateWithHole
from parametricpinn.types import Tensor

edge_length = 10.0
radius = 1.0
x_min = -edge_length
x_max = 0.0
y_min = 0.0
y_max = edge_length
volume_foce = torch.tensor([10.0, 10.0])
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
traction_left = torch.tensor([-100.0, 0.0])
traction_top = torch.tensor([0.0, 0.0])
traction_hole = torch.tensor([0.0, 0.0])
num_samples_per_parameter = 2
num_samples = num_samples_per_parameter**2
num_collocation_points = 3
num_points_per_bc = 3
num_symmetry_bcs = 2
num_points_symmetry_bcs = num_symmetry_bcs * num_points_per_bc
num_traction_bcs = 3
num_points_traction_bcs = num_traction_bcs * num_points_per_bc


### Test TrainingDataset2D
def _create_plate_with_hole_shape() -> Polygon:
    plate = box(x_min, x_max, y_min, y_max)
    hole = Point(0, 0).buffer(radius)
    return plate.difference(hole)


geometry = PlateWithHole(edge_length=edge_length, radius=radius)
shape = _create_plate_with_hole_shape()


@pytest.fixture
def sut() -> TrainingDataset2D:
    fake_geometry = geometry
    return TrainingDataset2D(
        geometry=fake_geometry,
        traction_left=traction_left,
        volume_force=volume_foce,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        num_samples_per_parameter=num_samples_per_parameter,
    )


def assert_if_coordinates_are_inside_shape(coordinates: Tensor) -> bool:
    coordinates_list = coordinates.tolist()
    for one_coordinate in coordinates_list:
        if not shape.contains(Point(one_coordinate[0], one_coordinate[1])):
            return False
    return True


def generate_expected_x_youngs_modulus(num_points: int):
    return [
        (0, torch.full((num_points, 1), min_youngs_modulus)),
        (1, torch.full((num_points, 1), min_youngs_modulus)),
        (2, torch.full((num_points, 1), max_youngs_modulus)),
        (3, torch.full((num_points, 1), max_youngs_modulus)),
    ]


def generate_expected_x_poissons_ratio(num_points: int):
    return [
        (0, torch.full((num_points, 1), min_poissons_ratio)),
        (1, torch.full((num_points, 1), max_poissons_ratio)),
        (2, torch.full((num_points, 1), min_poissons_ratio)),
        (3, torch.full((num_points, 1), max_poissons_ratio)),
    ]


def test_len(sut: TrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(sut: TrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_collocation_points),
)
def test_sample_pde__x_youngs_modulus(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_collocation_points),
)
def test_sample_pde__x_poissons_ratio(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_nu

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(sut: TrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_symmetry_bc__x_coordinates(
    sut: TrainingDataset2D, idx_sample: int
) -> None:
    _, sample_symmetry_bc, _ = sut[idx_sample]

    actual = sample_symmetry_bc.x_coor

    expected = torch.tensor(
        [
            [0.0, radius],
            [0.0, radius + 1 / 2 * (edge_length - radius)],
            [0.0, edge_length],
            [-edge_length, 0.0],
            [-radius - 1 / 2 * (edge_length - radius), 0.0],
            [-radius, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_symmetry_bcs),
)
def test_sample_symmetry_bc__x_youngs_modulus(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_symmetry_bc, _ = sut[idx_sample]

    actual = sample_symmetry_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_symmetry_bcs),
)
def test_sample_symmetry_bc__x_poissons_ratio(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_symmetry_bc, _ = sut[idx_sample]

    actual = sample_symmetry_bc.x_nu
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: TrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_coor

    expected = torch.tensor(
        [
            [-edge_length, 1 / 3 * edge_length],
            [-edge_length, 2 / 3 * edge_length],
            [-edge_length, edge_length],
            [-3 / 4 * edge_length, edge_length],
            [-2 / 4 * edge_length, edge_length],
            [-1 / 4 * edge_length, edge_length],
            [
                -torch.cos(torch.deg2rad(torch.tensor(1 / 4 * 90))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(1 / 4 * 90))) * radius,
            ],
            [
                -torch.cos(torch.deg2rad(torch.tensor(2 / 4 * 90))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(2 / 4 * 90))) * radius,
            ],
            [
                -torch.cos(torch.deg2rad(torch.tensor(3 / 4 * 90))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(3 / 4 * 90))) * radius,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_traction_bcs),
)
def test_sample_traction_bc__x_youngs_modulus(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_traction_bcs),
)
def test_sample_traction_bc__x_poissons_ratio(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_nu
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(sut: TrainingDataset2D, idx_sample: int) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.normal

    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [
                torch.cos(torch.deg2rad(torch.tensor(1 / 4 * 90))),
                -torch.sin(torch.deg2rad(torch.tensor(1 / 4 * 90))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(2 / 4 * 90))),
                -torch.sin(torch.deg2rad(torch.tensor(2 / 4 * 90))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(3 / 4 * 90))),
                -torch.sin(torch.deg2rad(torch.tensor(3 / 4 * 90))),
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__area_fractions(
    sut: TrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.area_frac

    hole_bc_length = 1 / 4 * 2.0 * math.pi * radius
    expected = torch.tensor(
        [
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(sut: TrainingDataset2D, idx_sample: int) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.y_true

    expected = torch.stack(
        (
            traction_left,
            traction_left,
            traction_left,
            traction_top,
            traction_top,
            traction_top,
            traction_hole,
            traction_hole,
            traction_hole,
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected)


### Test collate_training_data_2D()
@pytest.fixture
def fake_batch() -> (
    list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sample_collocation_0 = TrainingData2DCollocation(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_E=torch.tensor([[1.1]]),
        x_nu=torch.tensor([[1.2]]),
        f=torch.tensor([[1.3]]),
    )
    sample_symmetry_bc_0 = TrainingData2DSymmetryBC(
        x_coor=torch.tensor([[2.0, 2.0]]),
        x_E=torch.tensor([[2.1]]),
        x_nu=torch.tensor([[2.2]]),
    )
    sample_traction_bc_0 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[3.0, 3.0]]),
        x_E=torch.tensor([[3.1]]),
        x_nu=torch.tensor([[3.2]]),
        normal=torch.tensor([[3.3, 3.3]]),
        area_frac=torch.tensor([[3.4]]),
        y_true=torch.tensor([[3.5, 3.5]]),
    )
    sample_collocation_1 = TrainingData2DCollocation(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_E=torch.tensor([[10.1]]),
        x_nu=torch.tensor([[10.2]]),
        f=torch.tensor([[10.3]]),
    )
    sample_symmetry_bc_1 = TrainingData2DSymmetryBC(
        x_coor=torch.tensor([[20.0, 20.0]]),
        x_E=torch.tensor([[20.1]]),
        x_nu=torch.tensor([[20.2]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[30.0, 30.0]]),
        x_E=torch.tensor([[30.1]]),
        x_nu=torch.tensor([[30.2]]),
        normal=torch.tensor([[30.3, 30.3]]),
        area_frac=torch.tensor([[30.4]]),
        y_true=torch.tensor([[30.5, 30.5]]),
    )
    return [
        (sample_collocation_0, sample_symmetry_bc_0, sample_traction_bc_0),
        (sample_collocation_1, sample_symmetry_bc_1, sample_traction_bc_1),
    ]


def test_batch_pde__x_coordinate(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    batch_pde, _, _ = sut(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_youngs_modulus(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    batch_pde, _, _ = sut(fake_batch)
    actual = batch_pde.x_E

    expected = torch.tensor([[1.1], [10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_poissons_ratio(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    batch_pde, _, _ = sut(fake_batch)
    actual = batch_pde.x_nu

    expected = torch.tensor([[1.2], [10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    batch_pde, _, _ = sut(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.3], [10.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_coordinate(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, batch_symmetry_bc, _ = sut(fake_batch)
    actual = batch_symmetry_bc.x_coor

    expected = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_youngs_modulus(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, batch_symmetry_bc, _ = sut(fake_batch)
    actual = batch_symmetry_bc.x_E

    expected = torch.tensor([[2.1], [20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_poissons_ratio(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, batch_symmetry_bc, _ = sut(fake_batch)
    actual = batch_symmetry_bc.x_nu

    expected = torch.tensor([[2.2], [20.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_coordinate(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.x_coor

    expected = torch.tensor([[3.0, 3.0], [30.0, 30.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_youngs_modulus(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.x_E

    expected = torch.tensor([[3.1], [30.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_poissons_ratio(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.x_nu

    expected = torch.tensor([[3.2], [30.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__normal(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.normal

    expected = torch.tensor([[3.3, 3.3], [30.3, 30.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__area_fraction(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.area_frac

    expected = torch.tensor([[3.4], [30.4]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__y_true(
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DSymmetryBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sut = collate_training_data_2D

    _, _, batch_traction_bc = sut(fake_batch)
    actual = batch_traction_bc.y_true

    expected = torch.tensor([[3.5, 3.5], [30.5, 30.5]])
    torch.testing.assert_close(actual, expected)
