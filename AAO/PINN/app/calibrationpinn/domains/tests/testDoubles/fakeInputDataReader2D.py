# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.domains.inputreader import (
    ObservationData2D,
    ForceData2D,
    SolutionData2D,
)
from calibrationpinn.inputoutput import CSVDataReader


class FakeInputDataReader2D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        pass

    def read_observation_data(self) -> ObservationData2D:
        return ObservationData2D(
            coordinates=np.array([[0.0, 2.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0]]),
            displacements=np.array(
                [[30.0, 300.0], [10.0, 300.0], [30.0, 100.0], [10.0, 100.0]]
            ),
            youngs_moduli=np.array([[40.0], [20.0], [40.0], [20.0]]),
            poissons_ratios=np.array([[400.0], [400.0], [200.0], [200.0]]),
        )

    def read_force_data(self) -> ForceData2D:
        return ForceData2D(
            volume_force=np.array([4.0, 5.0]),
            coordinates_traction_bc=np.array([[7.0, 8.0]]),
            normals_traction_bc=np.array([[9.0, 10.0]]),
            traction_bc=np.array([[11.0]]),
        )

    def read_solution_data(self) -> SolutionData2D:
        return SolutionData2D(
            coordinates=np.array(
                [[0.0, 2.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0], [2.0, 0.0]]
            ),
            displacements=np.array(
                [
                    [31.0, 301.0],
                    [11.0, 301.0],
                    [21.0, 201.0],
                    [31.0, 101.0],
                    [11.0, 101.0],
                ]
            ),
            youngs_moduli=np.array([[41.0], [21.0], [31.0], [41.0], [21.0]]),
            poissons_ratios=np.array([[401.0], [401.0], [301.0], [201.0], [201.0]]),
        )
