# Standard library imports
from pathlib import Path
from typing import Optional

# Third-party imports

# Local library imports
from calibrationpinn.errors import DirectoryNotFoundError
from calibrationpinn.errors import FileNotFoundError
from calibrationpinn.settings import Settings


class PathAdministrator:
    def __init__(self, settings: Settings) -> None:
        self._project_dir_path = settings.PROJECT_DIRECTORY_PATH
        self._project_output_subdir_name = settings.OUTPUT_SUBDIRECTORY_NAME
        self._project_input_subdir_name = settings.INPUT_SUBDIRECTORY_NAME
        self._project_output_subdir_path: Path
        self._project_input_subdir_path: Path
        self._initialize_paths_to_project_subdirs()

    def get_path_to_output_file(
        self, output_file_name: str, output_subdir_name: Optional[str] = None
    ) -> Path:
        output_subdir_path = self._project_output_subdir_path
        if output_subdir_name is not None:
            output_subdir_path = self._join_path_to_subdir(
                self._project_output_subdir_path, output_subdir_name
            )
        return self._join_path_to_file(output_subdir_path, output_file_name)

    def get_path_to_existing_output_file(
        self, output_file_name: str, output_subdir_name: Optional[str] = None
    ) -> Path:
        output_subdir_path = self._project_output_subdir_path
        if output_subdir_name is not None:
            output_subdir_path = self._join_path_to_existing_subdir(
                self._project_output_subdir_path, output_subdir_name
            )
        return self._join_path_to_existing_file(output_subdir_path, output_file_name)

    def get_path_to_input_file(
        self, input_file_name: str, input_subdir_name: Optional[str] = None
    ) -> Path:
        input_subdir_path = self._project_input_subdir_path
        if input_subdir_name is not None:
            input_subdir_path = self._join_path_to_existing_subdir(
                self._project_input_subdir_path, input_subdir_name
            )
        return self._join_path_to_existing_file(input_subdir_path, input_file_name)

    def _initialize_paths_to_project_subdirs(self) -> None:
        self._project_output_subdir_path = self._join_path_to_project_subdir(
            self._project_output_subdir_name
        )
        self._project_input_subdir_path = self._join_path_to_project_subdir(
            self._project_input_subdir_name
        )

    def _join_path_to_project_subdir(self, project_subdir_name: str) -> Path:
        return self._join_path_to_subdir(self._project_dir_path, project_subdir_name)

    def _join_path_to_subdir(self, dir_path: Path, subdir_name: str) -> Path:
        subdir_path = dir_path / subdir_name
        if not Path.is_dir(subdir_path):
            Path.mkdir(subdir_path, parents=True)
        return subdir_path

    def _join_path_to_existing_subdir(
        self, existing_dir_path: Path, subdir_name: str
    ) -> Path:
        subdir_path = existing_dir_path / subdir_name
        if not Path.is_dir(subdir_path):
            raise DirectoryNotFoundError(subdir_path)
        return subdir_path

    def _join_path_to_file(self, dir_path: Path, file_name: str) -> Path:
        if not Path.is_dir(dir_path):
            raise DirectoryNotFoundError(dir_path)
        return dir_path / file_name

    def _join_path_to_existing_file(
        self, existing_dir_path: Path, file_name: str
    ) -> Path:
        file_path = existing_dir_path / file_name
        if not Path.is_file(file_path):
            raise FileNotFoundError(file_path)
        return file_path
