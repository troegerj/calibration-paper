# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader
from calibrationpinn.typeAliases import NPArray


class FakeSolutionData1D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        self._data_reader = data_reader
        self.coordinates: NPArray = np.array([[0.0]])
        self.displacements: NPArray = np.array([[0.0]])
        self.youngs_moduli: NPArray = np.array([[0.0]])
