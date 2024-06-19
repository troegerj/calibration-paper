# Standard library imports
from typing import Optional

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader


class FakeSolutionData2D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        self._data_reader = data_reader
        self.coordinates = jnp.array([[0.0, 0.0]])
        self.displacements = jnp.array([[0.0, 0.0]])
        self.youngs_moduli = jnp.array([[0.0]])
        self.poissons_ratios = jnp.array([[0.0]])
