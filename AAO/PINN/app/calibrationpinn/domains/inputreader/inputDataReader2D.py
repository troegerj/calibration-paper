# Standard library imports
from typing import NamedTuple, Optional, Protocol

# Third-party imports

# Local library imports
from calibrationpinn.inputoutput import CSVDataReader
from calibrationpinn.typeAliases import NPArray


class ObservationData2D(NamedTuple):
    coordinates: NPArray
    displacements: NPArray
    youngs_moduli: NPArray
    poissons_ratios: NPArray


class ForceData2D(NamedTuple):
    volume_force: NPArray
    coordinates_traction_bc: NPArray
    normals_traction_bc: NPArray
    traction_bc: NPArray


class SolutionData2D(NamedTuple):
    coordinates: NPArray
    displacements: NPArray
    youngs_moduli: NPArray
    poissons_ratios: NPArray


class InputDataReader2DProtocol(Protocol):
    def read_observation_data(self) -> ObservationData2D:
        pass

    def read_force_data(self) -> ForceData2D:
        pass

    def read_solution_data(self) -> SolutionData2D:
        pass


class InputDataReader2D:
    def __init__(
        self, data_reader: CSVDataReader, input_subdir_name: Optional[str] = None
    ) -> None:
        class _ObservationAndSolutionFileColumns:
            def __init__(self):
                self.coordinates = slice(0, 2)
                self.displacements = slice(2, 4)
                self.youngs_moduli = 4
                self.poissons_ratio = 5

        class _TractionBCFileColumns:
            def __init__(self):
                self.coordinates = slice(0, 2)
                self.normals = slice(2, 4)
                self.traction_bc = slice(4, 6)

        self._data_reader = data_reader
        self._input_subdir_name = input_subdir_name
        self._file_name_observation = "observation"
        self._columns_observation = _ObservationAndSolutionFileColumns()
        self._file_name_volume_force = "volume_force"
        self._file_name_traction_bc = "traction_bc"
        self._columns_traction_bc = _TractionBCFileColumns()
        self._file_name_solution = "solution"
        self._columns_solution = _ObservationAndSolutionFileColumns()

    def read_observation_data(self) -> ObservationData2D:
        observation = self._data_reader.read(
            self._file_name_observation, self._input_subdir_name
        )
        return ObservationData2D(
            coordinates=observation[:, self._columns_observation.coordinates],
            displacements=observation[:, self._columns_observation.displacements],
            youngs_moduli=observation[
                :, self._columns_observation.youngs_moduli
            ].reshape((-1, 1)),
            poissons_ratios=observation[
                :, self._columns_observation.poissons_ratio
            ].reshape((-1, 1)),
        )

    def read_force_data(self) -> ForceData2D:
        volume_force = self._data_reader.read(
            self._file_name_volume_force, self._input_subdir_name
        )
        traction_bc = self._data_reader.read(
            self._file_name_traction_bc, self._input_subdir_name
        )
        return ForceData2D(
            volume_force=volume_force[0],
            coordinates_traction_bc=traction_bc[
                :, self._columns_traction_bc.coordinates
            ],
            normals_traction_bc=traction_bc[:, self._columns_traction_bc.normals],
            traction_bc=traction_bc[:, self._columns_traction_bc.traction_bc],
        )

    def read_solution_data(self) -> SolutionData2D:
        solution = self._data_reader.read(
            self._file_name_solution, self._input_subdir_name
        )
        return SolutionData2D(
            coordinates=solution[:, self._columns_solution.coordinates],
            displacements=solution[:, self._columns_solution.displacements],
            youngs_moduli=solution[:, self._columns_solution.youngs_moduli].reshape(
                (-1, 1)
            ),
            poissons_ratios=solution[:, self._columns_solution.poissons_ratio].reshape(
                (-1, 1)
            ),
        )
