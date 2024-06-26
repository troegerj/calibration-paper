# Standard library imports
from pathlib import Path
from typing import Type

# Third-party imports

# Local library imports


class Error(Exception):
    pass


class DirectoryNotFoundError(Error):
    def __init__(self, path_to_directory: Path) -> None:
        self._message = "The directory {path_to_directory} could not be found".format(
            path_to_directory=path_to_directory
        )
        super().__init__(self._message)


class FileNotFoundError(Error):
    def __init__(self, path_to_file: Path) -> None:
        self._message = "The requested file {path_to_file} could not be found!".format(
            path_to_file=path_to_file
        )
        super().__init__(self._message)


class TypeNotSupportedError(Error):
    def __init__(self, type: Type) -> None:
        self._message = "Type {type} not supported!".format(type=type)
        super().__init__(self._message)
