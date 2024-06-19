# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.domains.inputreader import (
    ObservationData1D,
    ForceData1D,
    SolutionData1D,
)
from calibrationpinn.inputoutput import CSVDataReader


class FakeInputDataReader1D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        pass

    def read_observation_data(self) -> ObservationData1D:
        return ObservationData1D(
            coordinates=np.array([[0.0], [0.0]]),
            displacements=np.array([[1.0], [1.0]]),
            youngs_moduli=np.array([[2.0], [2.0]]),
        )

    def read_force_data(self) -> ForceData1D:
        return ForceData1D(
            volume_force=np.array([3.0]),
            coordinates_traction_bc=np.array([[4.0]]),
            traction_bc=np.array([[5.0]]),
        )

    def read_solution_data(self) -> SolutionData1D:
        return SolutionData1D(
            coordinates=np.array([[6.0], [6.0]]),
            displacements=np.array([[7.0], [7.0]]),
            youngs_moduli=np.array([[8.0], [8.0]]),
        )
