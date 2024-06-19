# Standard library imports
from pathlib import Path
import shutil
import unittest

# Third-party imports

# Local library imports
from calibrationpinn.assertions import assert_equal, assert_raises_error
from calibrationpinn.errors import DirectoryNotFoundError, FileNotFoundError
from calibrationpinn.inputoutput.pathAdministrator import PathAdministrator
from calibrationpinn.settings import Settings


class TestPathAdministrator(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._settings = Settings()
        test_project_directory_path = (
            Path() / "calibrationpinn" / "inputoutput" / "tests"
        )
        self._settings.PROJECT_DIRECTORY_PATH = test_project_directory_path
        self._settings.INPUT_SUBDIRECTORY_NAME = "testInput"
        self._settings.OUTPUT_SUBDIRECTORY_NAME = "testOutput"
        self._sut = PathAdministrator(self._settings)

    # Path to output file
    def test_get_path_to_output_file_without_additional_output_subdirectory(
        self,
    ) -> None:
        """
        Test that the path to the output file is returned correctly.
        """

        output_file_name = "new_output_file.txt"

        actual = self._sut.get_path_to_output_file(output_file_name=output_file_name)

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.OUTPUT_SUBDIRECTORY_NAME
            / output_file_name
        )
        assert_equal(self, expected=expected, actual=actual)

    def test_get_path_to_output_file_with_additional_output_subdirectory(self) -> None:
        """
        Test that the path to the output file is returned correctly.
        """

        output_file_name = "new_output_file.txt"
        output_subdirectory_name = "new_output_subdirectory"

        actual = self._sut.get_path_to_output_file(
            output_file_name=output_file_name,
            output_subdir_name=output_subdirectory_name,
        )

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.OUTPUT_SUBDIRECTORY_NAME
            / output_subdirectory_name
            / output_file_name
        )
        shutil.rmtree(Path(str(actual).replace("/" + output_file_name, "")))
        assert_equal(self, expected=expected, actual=actual)

    # Path to existing output file
    def test_get_path_to_existing_output_file_without_additional_output_subdirectory(
        self,
    ) -> None:
        """
        Test that the path to the existing output file is returned correctly.
        """

        output_file_name = "output_file.txt"

        actual = self._sut.get_path_to_existing_output_file(
            output_file_name=output_file_name
        )

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.OUTPUT_SUBDIRECTORY_NAME
            / output_file_name
        )
        assert_equal(self, expected=expected, actual=actual)

    def test_get_path_to_existing_output_file_with_additional_output_subdirectory(
        self,
    ) -> None:
        """
        Test that the path to the existing output file is returned correctly.
        """

        output_file_name = "output_file.txt"
        output_subdirectory_name = "output_subdirectory"

        actual = self._sut.get_path_to_existing_output_file(
            output_file_name=output_file_name,
            output_subdir_name=output_subdirectory_name,
        )

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.OUTPUT_SUBDIRECTORY_NAME
            / output_subdirectory_name
            / output_file_name
        )
        assert_equal(self, expected=expected, actual=actual)

    # Path to input file
    def test_get_path_to_input_file_without_additional_input_subdirectory(self) -> None:
        """
        Test that the path to the input file is returned correctly.
        """

        input_file_name = "input_file.txt"

        actual = self._sut.get_path_to_input_file(input_file_name=input_file_name)

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.INPUT_SUBDIRECTORY_NAME
            / input_file_name
        )
        assert_equal(self, expected=expected, actual=actual)

    def test_get_path_to_input_file_with_additional_input_subdirectory(self) -> None:
        """
        Test that the path to the input file is returned correctly.
        """

        input_file_name = "input_file.txt"
        input_subdirectory_name = "input_subdirectory"

        actual = self._sut.get_path_to_input_file(
            input_file_name=input_file_name, input_subdir_name=input_subdirectory_name
        )

        expected = (
            Path(self._settings.PROJECT_DIRECTORY_PATH)
            / self._settings.INPUT_SUBDIRECTORY_NAME
            / input_subdirectory_name
            / input_file_name
        )
        assert_equal(self, expected=expected, actual=actual)

    # Errors
    def test_get_path_to_existing_output_file_for_not_existing_subdirectory(
        self,
    ) -> None:
        """
        Test that a DirectoryNotFoundError error is thrown if the output subdirectory does not exist.
        """
        output_file_name = "output_file"
        output_subdirectory_name = "not_existing_output_subdirectory"
        assert_raises_error(
            self,
            DirectoryNotFoundError,
            self._sut.get_path_to_existing_output_file,
            output_file_name,
            output_subdirectory_name,
        )

    def test_get_path_to_existing_output_file_for_not_existing_output_file(
        self,
    ) -> None:
        """
        Test that a FileNotFoundError error is thrown if the output file does not exist.
        """
        output_file_name = "not_existing_output_file"
        output_subdirectory_name = "output_subdirectory"
        assert_raises_error(
            self,
            FileNotFoundError,
            self._sut.get_path_to_existing_output_file,
            output_file_name,
            output_subdirectory_name,
        )

    def test_get_path_to_input_file_for_not_existing_subdirectory(self) -> None:
        """
        Test that a DirectoryNotFoundError error is thrown if the input subdirectory does not exist.
        """
        input_file_name = "input_file"
        input_subdirectory_name = "not_existing_input_subdirectory"
        assert_raises_error(
            self,
            DirectoryNotFoundError,
            self._sut.get_path_to_input_file,
            input_file_name,
            input_subdirectory_name,
        )

    def test_get_path_to_input_file_for_not_existing_input_file(self) -> None:
        """
        Test that a FileNotFoundError error is thrown if the input file does not exist.
        """
        input_file_name = "not_existing_input_file"
        input_subdirectory_name = "input_subdirectory"
        assert_raises_error(
            self,
            FileNotFoundError,
            self._sut.get_path_to_input_file,
            input_file_name,
            input_subdirectory_name,
        )
