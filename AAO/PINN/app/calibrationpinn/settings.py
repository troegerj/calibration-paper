# Standard library imports
import os
from pathlib import Path

# Third-party imports
import jax
import numpy as np

# Local library imports


# Floating point number precision
jax.config.update("jax_enable_x64", True)


class Settings:
    def __init__(self) -> None:
        self.PROJECT_DIRECTORY_PATH = Path(
            os.getenv("CALIBRATIONPINN_HOME", os.getenv("HOME", "."))
        )
        self.OUTPUT_SUBDIRECTORY_NAME = "output"
        self.INPUT_SUBDIRECTORY_NAME = "input"

        self.DATA_TYPE = np.float64
