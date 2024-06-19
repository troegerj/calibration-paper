# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader


class FakeForceData1D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        self.volume_force = np.array([0.0])
        self.coordinates_traction_bc = np.array([[0.0]])
        self.traction_bc = np.array([[0.0]])
