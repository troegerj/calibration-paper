# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd

# Local library imports
from calibrationpinn import Settings
from calibrationpinn.inputoutput import (
    DATDataReader,
    PandasDataWriter,
    PathAdministrator,
)

youngs_modulus_least_squares = 222702.0482  # N/mm2
traction_bc_free_end = 212.5524  # N/mm2
volume_force = 0.0  # N/mm3


settings = Settings()
settings.OUTPUT_SUBDIRECTORY_NAME = settings.INPUT_SUBDIRECTORY_NAME
path_admin = PathAdministrator(settings)
data_reader = DATDataReader(path_admin)
data_writer = PandasDataWriter(path_admin)

input_subdir_name = output_subdir_name = os.path.join(
    "Paper_PINNs_Anton", "1D_Experimental"
)
input_file_name_DIC_data = "displacements.dat"

output_file_name_observation = "observation"
output_file_name_volume_force = "volume_force"
output_file_name_traction_bc = "traction_bc"

header_observation = ["x [mm]", "displacement_x [mm]", "Young's modulus [N/mm2]"]
header_volume_force = ["volume_force [N/mm3]"]
header_traction_bc = ["x [mm]", "traction [N/mm2]"]


def preprocess_SteelTS275_data_1D():
    print("Preprocess input data ...")
    dic_data = data_reader.read(input_file_name_DIC_data, input_subdir_name)
    coordinates = dic_data[:, 0].reshape((-1, 1))
    displacements = dic_data[:, 1].reshape((-1, 1))

    # observation
    youngs_moduli = np.full((displacements.shape[0], 1), youngs_modulus_least_squares)
    observation = np.hstack((coordinates, displacements, youngs_moduli))
    data_writer.write(
        pd.DataFrame(observation),
        output_file_name_observation,
        output_subdir_name,
        header_observation,
    )

    # volume force
    data_writer.write(
        pd.DataFrame(np.array([volume_force])),
        output_file_name_volume_force,
        output_subdir_name,
        header_volume_force,
    )

    # traction bc
    traction_bc_left_coordinate = coordinates[0, 0]
    traction_bc_right_coordinate = coordinates[-1, 0]
    traction_bc_left_traction = -traction_bc_free_end
    traction_bc_right_traction = traction_bc_free_end
    traction_bc = np.array(
        [
            [traction_bc_left_coordinate, traction_bc_left_traction],
            [traction_bc_right_coordinate, traction_bc_right_traction],
        ]
    )
    data_writer.write(
        pd.DataFrame(traction_bc),
        output_file_name_traction_bc,
        output_subdir_name,
        header_traction_bc,
    )
