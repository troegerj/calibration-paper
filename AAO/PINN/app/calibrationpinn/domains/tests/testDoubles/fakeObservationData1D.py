# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader


class FakeObservationData1D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        self.coordinates = np.array([[0.0]])
        self.displacements = np.array([[0.0]])
        self.youngs_moduli = np.array([[0.0]])
