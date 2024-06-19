# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.domains.dataselection import DataSelectionFuncProtocol
from calibrationpinn.typeAliases import JNPArray, PRNGKey


class FakeDataSelectionFunc(DataSelectionFuncProtocol):
    def __call__(
        self, data: JNPArray, num_data_points: int, PRNG_key: PRNGKey
    ) -> JNPArray:
        return data
