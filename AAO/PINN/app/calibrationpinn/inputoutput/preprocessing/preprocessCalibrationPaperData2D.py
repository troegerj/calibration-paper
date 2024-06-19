# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd

# Local library imports
from calibrationpinn import Settings
from calibrationpinn.inputoutput import (
    CSVDataReader,
    PandasDataWriter,
    PathAdministrator,
)

### Predefined
youngs_modulus = 210000  # N/mm2
poissons_ratio = 0.3
traction_bc_left_edge_x = -150  # N/mm2
traction_bc_left_edge_y = 0  # N/mm2
traction_bc_top_edge_x = 0  # N/mm2
traction_bc_top_edge_y = 0  # N/mm2
traction_bc_hole_x = 0  # N/mm2
traction_bc_hole_y = 0  # N/mm2
volume_force = np.array([0.0, 0.0])  # N/mm3
length_edge = 10
radius = 2
### Configuration
num_points_traction_bc_edge = 64
num_points_traction_bc_hole = 32

settings = Settings()
settings.OUTPUT_SUBDIRECTORY_NAME = settings.INPUT_SUBDIRECTORY_NAME
path_admin = PathAdministrator(settings)
data_reader = CSVDataReader(path_admin)
data_writer = PandasDataWriter(path_admin)


output_file_name_observation = "observation"
output_file_name_volume_force = "volume_force"
output_file_name_traction_bc = "traction_bc"

header_observation = [
    "x [mm]",
    "y [mm]",
    "displacement_x [mm]",
    "displacement_y [mm]",
    "Young's modulus [N/mm2]",
    "Poisson's ratio [-]",
]
header_volume_force = ["volume_force_x [N/mm3]", "volume_force_y [N/mm3]"]
header_traction_bc = [
    "x [mm]",
    "y [mm]",
    "normal_x",
    "normal_y",
    "traction_x [N/mm2]",
    "traction_y [N/mm2]",
]


def preprocess_calibration_paper_data_2D(input_file_name, input_subdir_name):
    print("Preprocess input data ...")
    output_subdir_name = input_subdir_name
    data = data_reader.read(input_file_name, input_subdir_name)
    coordinates_x = data[:, 0].reshape((-1, 1))
    coordinates_y = data[:, 1].reshape((-1, 1))
    displacements_x = data[:, 2].reshape((-1, 1))
    displacements_y = data[:, 3].reshape((-1, 1))

    ### observation
    youngs_moduli = np.full((displacements_x.shape[0], 1), youngs_modulus)
    poissons_ratios = np.full((displacements_x.shape[0], 1), poissons_ratio)
    observation = np.hstack(
        (
            coordinates_x,
            coordinates_y,
            displacements_x,
            displacements_y,
            youngs_moduli,
            poissons_ratios,
        )
    )
    data_writer.write(
        pd.DataFrame(observation),
        output_file_name_observation,
        output_subdir_name,
        header_observation,
    )

    ### volume force
    data_writer.write(
        pd.DataFrame(np.array([volume_force])),
        output_file_name_volume_force,
        output_subdir_name,
        header_volume_force,
    )

    ### traction bc
    # left edge
    left_bc_coordinates_x = np.full((num_points_traction_bc_edge, 1), 0.0)
    left_bc_coordinates_y = np.linspace(
        0.0, length_edge, num_points_traction_bc_edge, endpoint=False
    ).reshape((-1, 1))

    left_bc_coordinates = np.hstack((left_bc_coordinates_x, left_bc_coordinates_y))
    left_bc_traction = np.tile(
        np.array([traction_bc_left_edge_x, traction_bc_left_edge_y]),
        (num_points_traction_bc_edge, 1),
    )
    left_bc_normal = np.array([-1, 0])
    left_bc_nomals = np.tile(left_bc_normal, (num_points_traction_bc_edge, 1))

    # top edge
    top_bc_coordinates_x = np.linspace(
        0.0, length_edge, num_points_traction_bc_edge
    ).reshape((-1, 1))
    top_bc_coordinates_y = np.full((num_points_traction_bc_edge, 1), length_edge)
    top_bc_coordinates = np.hstack((top_bc_coordinates_x, top_bc_coordinates_y))
    top_bc_traction = np.tile(
        np.array([traction_bc_top_edge_x, traction_bc_top_edge_y]),
        (num_points_traction_bc_edge, 1),
    )
    top_bc_normal = np.array([0, 1])
    top_bc_normals = np.tile(top_bc_normal, (num_points_traction_bc_edge, 1))

    # hole
    angle = np.linspace(0, 90, num_points_traction_bc_hole).reshape((-1, 1))
    delta_x = np.cos(np.radians(angle)) * radius
    delta_y = np.sin(np.radians(angle)) * radius
    hole_bc_coordinates_x = length_edge - delta_x
    hole_bc_coordinates_y = delta_y
    hole_bc_coordinates = np.hstack((hole_bc_coordinates_x, hole_bc_coordinates_y))
    hole_bc_traction = np.tile(
        np.array([traction_bc_hole_x, traction_bc_hole_y]),
        (num_points_traction_bc_hole, 1),
    )
    hole_bc_normals_length = np.sqrt(
        hole_bc_coordinates_x**2 + hole_bc_coordinates_y**2
    )
    hole_bc_normals = np.hstack((delta_x, -delta_y)) / hole_bc_normals_length

    # combined coordinates, normals and tractions
    bc_coordinates = np.vstack(
        (left_bc_coordinates, top_bc_coordinates, hole_bc_coordinates)
    )
    bc_normals = np.vstack((left_bc_nomals, top_bc_normals, hole_bc_normals))
    bc_traction = np.vstack((left_bc_traction, top_bc_traction, hole_bc_traction))

    bc_coordinates_x = bc_coordinates[:, 0].reshape((-1, 1))
    bc_coordinates_y = bc_coordinates[:, 1].reshape((-1, 1))
    bc_traction_x = bc_traction[:, 0].reshape((-1, 1))
    bc_traction_y = bc_traction[:, 1].reshape((-1, 1))
    bc_normals_x = bc_normals[:, 0].reshape((-1, 1))
    bc_normals_y = bc_normals[:, 1].reshape((-1, 1))

    traction_bc = np.hstack(
        (
            bc_coordinates_x,
            bc_coordinates_y,
            bc_normals_x,
            bc_normals_y,
            bc_traction_x,
            bc_traction_y,
        )
    )
    data_writer.write(
        pd.DataFrame(traction_bc),
        output_file_name_traction_bc,
        output_subdir_name,
        header_traction_bc,
    )
