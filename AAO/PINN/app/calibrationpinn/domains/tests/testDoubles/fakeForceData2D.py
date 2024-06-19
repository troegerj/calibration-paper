# Standard library imports
from typing import Optional

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader


class FakeForceData2D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        self.volume_force = jnp.array([0.0, 0.0])
        self.coordinates_traction_bc = jnp.array([[0.0, 0.0]])
        self.normals_traction_bc = jnp.array([[0.0, 0.0]])
        self.traction_bc = jnp.array([[0.0, 0.0]])
