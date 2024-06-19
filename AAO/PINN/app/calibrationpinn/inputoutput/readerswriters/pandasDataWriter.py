# Standard library imports
from typing import Union

# Third library imports

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.typeAliases import NPArray, PDDataFrame


class PandasDataWriter:
    def __init__(self, path_admin: PathAdministrator) -> None:
        self._path_admin = path_admin
        self._correct_file_ending = ".csv"

    def write(
        self,
        data: PDDataFrame,
        output_file_name: str,
        output_subdir_name: str,
        header: Union[bool, list[str]] = False,
        index: bool = False,
    ) -> None:
        output_file_name = self._ensure_correct_file_ending(output_file_name)
        output_file_path = self._path_admin.get_path_to_output_file(
            output_file_name, output_subdir_name
        )
        data.to_csv(output_file_path, header=header, index=index)

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending
