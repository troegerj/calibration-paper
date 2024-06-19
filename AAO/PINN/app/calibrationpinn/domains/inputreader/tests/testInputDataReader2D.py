# Standard library imports
import os
from pathlib import Path
import unittest


# Third-party imports
import numpy as np
import pandas as pd


# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains.inputreader import InputDataReader2D
from calibrationpinn.inputoutput import PathAdministrator, CSVDataReader
from calibrationpinn.settings import Settings


class TestInputDataReader2D(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        settings = Settings()
        self._test_project_dir_path = (
            Path() / "calibrationpinn" / "domains" / "inputdata" / "tests"
        )
        self._test_input_project_subdir_name = os.path.join("testInput", "data1D")
        self._test_output_project_subdir_name = os.path.join("testOutput", "data1D")
        self._test_input_subdir_path = os.path.join(
            self._test_project_dir_path,
            self._test_input_project_subdir_name,
        )
        settings.PROJECT_DIRECTORY_PATH = self._test_project_dir_path
        settings.OUTPUT_SUBDIRECTORY_NAME = self._test_output_project_subdir_name
        settings.INPUT_SUBDIRECTORY_NAME = self._test_input_project_subdir_name
        path_admin = PathAdministrator(settings)
        data_reader = CSVDataReader(path_admin)
        self._name_observation = "observation"
        self._coordinates_observation = np.array([[1.0, 1.0], [1.0, 1.0]])
        self._displacements_observation = np.array([[2.0, 2.0], [2.0, 2.0]])
        self._youngs_moduli_observation = np.array([[3.0], [3.0]])
        self._poissons_ratios_observation = np.array([[4.0], [4.0]])
        self._name_volume_force = "volume_force"
        self._volume_force = np.array([5.0, 5.0])
        self._name_traction_bc = "traction_bc"
        self._coordinates_traction_bc = np.array([[6.0, 6.0], [6.0, 6.0]])
        self._normals_traction_bc = np.array([[7.0, 7.0], [8.0, 8.0]])
        self._traction_bc = np.array([[9.0, 9.0], [9.0, 9.0]])
        self._name_solution = "solution"
        self._coordinates_solution = np.array([[10.0, 10.0], [10.0, 10.0]])
        self._displacements_solution = np.array([[11.0, 11.0], [11.0, 11.0]])
        self._youngs_moduli_solution = np.array([[12.0], [12.0]])
        self._poissons_ratios_solution = np.array([[13.0], [13.0]])
        self._set_up_test_csv_input_files()
        self._sut = InputDataReader2D(data_reader=data_reader, input_subdir_name=None)

    def tearDown(self) -> None:
        os.remove(self._join_output_path(self._name_observation))
        os.remove(self._join_output_path(self._name_volume_force))
        os.remove(self._join_output_path(self._name_traction_bc))
        os.remove(self._join_output_path(self._name_solution))

    # observation
    def test_observation_coordinates(self) -> None:
        """
        Test that the observation coordinates are read correctly.
        """
        observation_data = self._sut.read_observation_data()
        actual = observation_data.coordinates

        expected = self._coordinates_observation
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_observation_displacements(self) -> None:
        """
        Test that the observation displacements are read correctly.
        """
        observation_data = self._sut.read_observation_data()
        actual = observation_data.displacements

        expected = self._displacements_observation
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_observation_youngs_moduli(self) -> None:
        """
        Test that the observation Young's moduli are read correctly.
        """
        observation_data = self._sut.read_observation_data()
        actual = observation_data.youngs_moduli

        expected = self._youngs_moduli_observation
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_observation_poissons_ratios(self) -> None:
        """
        Test that the observation Poisson's ratios are read correctly.
        """
        observation_data = self._sut.read_observation_data()
        actual = observation_data.poissons_ratios

        expected = self._poissons_ratios_observation
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # force
    def test_volume_force(self) -> None:
        """
        Test that the volume force is read correctly.
        """
        force_data = self._sut.read_force_data()
        actual = force_data.volume_force

        expected = self._volume_force
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_coordinates_traction_bc(self) -> None:
        """
        Test that the traction boundary condition coordinates are read correctly.
        """
        force_data = self._sut.read_force_data()
        actual = force_data.coordinates_traction_bc

        expected = self._coordinates_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_normals_traction_bc(self) -> None:
        """
        Test that the traction boundary condition normals are read correctly.
        """
        force_data = self._sut.read_force_data()
        actual = force_data.normals_traction_bc

        expected = self._normals_traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_traction_bc(self) -> None:
        """
        Test that the traction boundary conditions are read correctly.
        """
        force_data = self._sut.read_force_data()
        actual = force_data.traction_bc

        expected = self._traction_bc
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    # solution
    def test_solution_coordinates(self) -> None:
        """
        Test that the solution coordinates are read correctly.
        """
        solution_data = self._sut.read_solution_data()
        actual = solution_data.coordinates

        expected = self._coordinates_solution
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_solution_displacements(self) -> None:
        """
        Test that the solution displacements are read correctly.
        """
        solution_data = self._sut.read_solution_data()
        actual = solution_data.displacements

        expected = self._displacements_solution
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_solution_youngs_moduli(self) -> None:
        """
        Test that the solution Young's moduli are read correctly.
        """
        solution_data = self._sut.read_solution_data()
        actual = solution_data.youngs_moduli

        expected = self._youngs_moduli_solution
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_solution_poissons_ratios(self) -> None:
        """
        Test that the solution Poisson's ratios are read correctly.
        """
        solution_data = self._sut.read_solution_data()
        actual = solution_data.poissons_ratios

        expected = self._poissons_ratios_solution
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def _set_up_test_csv_input_files(self) -> None:
        observation = np.hstack(
            (
                self._coordinates_observation,
                self._displacements_observation,
                self._youngs_moduli_observation,
                self._poissons_ratios_observation,
            )
        )
        header_observation = ["x", "y", "ux", "uy", "E", "nu"]
        pd.DataFrame(observation).to_csv(
            self._join_output_path(self._name_observation),
            header=header_observation,
            index=False,
        )
        volume_force = np.array([self._volume_force])
        header_volume_force = ["vfx", "vfy"]
        pd.DataFrame(volume_force).to_csv(
            self._join_output_path(self._name_volume_force),
            header=header_volume_force,
            index=False,
        )
        traction_bc = np.hstack(
            (
                self._coordinates_traction_bc,
                self._normals_traction_bc,
                self._traction_bc,
            )
        )
        header_traction_bc = ["x", "y", "nx", "ny", "tx", "ty"]
        pd.DataFrame(traction_bc).to_csv(
            self._join_output_path(self._name_traction_bc),
            header=header_traction_bc,
            index=False,
        )
        solution = np.hstack(
            (
                self._coordinates_solution,
                self._displacements_solution,
                self._youngs_moduli_solution,
                self._poissons_ratios_solution,
            )
        )
        header_solution = ["x", "y", "ux", "uy", "E", "nu"]
        pd.DataFrame(solution).to_csv(
            self._join_output_path(self._name_solution),
            header=header_solution,
            index=False,
        )

    def _join_output_path(self, file_name: str) -> str:
        file_name_with_ending = file_name + ".csv"
        return os.path.join(self._test_input_subdir_path, file_name_with_ending)
