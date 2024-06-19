# Standard library imports
from typing import NamedTuple, Optional, Protocol

# Third-party imports

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader
from calibrationpinn.typeAliases import NPArray


class ObservationData1D(NamedTuple):
    coordinates: NPArray
    displacements: NPArray
    youngs_moduli: NPArray


class ForceData1D(NamedTuple):
    volume_force: NPArray
    coordinates_traction_bc: NPArray
    traction_bc: NPArray


class SolutionData1D(NamedTuple):
    coordinates: NPArray
    displacements: NPArray
    youngs_moduli: NPArray


class InputDataReader1DProtocol(Protocol):
    def read_observation_data(self) -> ObservationData1D:
        pass

    def read_force_data(self) -> ForceData1D:
        pass

    def read_solution_data(self) -> SolutionData1D:
        pass


class InputDataReader1D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        class _ObservationAndSolutionFileColumns:
            def __init__(self):
                self.coordinates = 0
                self.displacements = 1
                self.youngs_moduli = 2

        class _TractionBCFileColumns:
            def __init__(self):
                self.coordinates = 0
                self.traction_bc = 1

        self._data_reader = data_reader
        self._input_subdir_name = input_subdir_name
        self._file_name_observation = "observation"
        self._columns_observation = _ObservationAndSolutionFileColumns()
        self._file_name_volume_force = "volume_force"
        self._file_name_traction_bc = "traction_bc"
        self._columns_traction_bc = _TractionBCFileColumns()
        self._file_name_solution = "solution"
        self._columns_solution = _ObservationAndSolutionFileColumns()

    def read_observation_data(self) -> ObservationData1D:
        observation = self._data_reader.read(
            self._file_name_observation, self._input_subdir_name
        )
        return ObservationData1D(
            coordinates=observation[:, self._columns_observation.coordinates].reshape(
                (-1, 1)
            ),
            displacements=observation[
                :, self._columns_observation.displacements
            ].reshape((-1, 1)),
            youngs_moduli=observation[
                :, self._columns_observation.youngs_moduli
            ].reshape((-1, 1)),
        )

    def read_force_data(self) -> ForceData1D:
        volume_force = self._data_reader.read(
            self._file_name_volume_force, self._input_subdir_name
        )
        traction_bc = self._data_reader.read(
            self._file_name_traction_bc, self._input_subdir_name
        )
        return ForceData1D(
            volume_force=volume_force[0],  # reduce 2D array to 1D array
            coordinates_traction_bc=traction_bc[
                :, self._columns_traction_bc.coordinates
            ].reshape((-1, 1)),
            traction_bc=traction_bc[:, self._columns_traction_bc.traction_bc].reshape(
                (-1, 1)
            ),
        )

    def read_solution_data(self) -> SolutionData1D:
        solution = self._data_reader.read(
            self._file_name_solution, self._input_subdir_name
        )
        return SolutionData1D(
            coordinates=solution[:, self._columns_solution.coordinates].reshape(
                (-1, 1)
            ),
            displacements=solution[:, self._columns_solution.displacements].reshape(
                (-1, 1)
            ),
            youngs_moduli=solution[:, self._columns_solution.youngs_moduli].reshape(
                (-1, 1)
            ),
        )
