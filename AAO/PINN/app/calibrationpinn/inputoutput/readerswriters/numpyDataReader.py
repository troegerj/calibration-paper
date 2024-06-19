# Standard library imports
from typing import Optional, Literal

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.typeAliases import NPArray


class NumpyDataReader:
    def __init__(self, path_admin: PathAdministrator) -> None:
        self._path_admin = path_admin
        self._correct_file_ending = ".npy"

    def read(
        self,
        input_file_name: str,
        allow_pickle: bool,
        encoding: Literal["ASCII", "latin1", "bytes"],
        input_subdir_name: Optional[str] = None,
    ) -> NPArray:
        input_file_name = self._ensure_correct_file_ending(input_file_name)
        input_file_path = self._path_admin.get_path_to_input_file(
            input_file_name, input_subdir_name
        )
        return np.load(input_file_path, allow_pickle=allow_pickle, encoding=encoding)

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending
