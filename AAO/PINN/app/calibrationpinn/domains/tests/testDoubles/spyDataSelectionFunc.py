# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.domains.dataselection import DataSelectionFuncProtocol
from calibrationpinn.typeAliases import JNPArray, PRNGKey


class SpyDataSelectionFunc(DataSelectionFuncProtocol):
    def __init__(self) -> None:
        self.num_points: list[int] = []
        self.PRNG_key: PRNGKey

    def __call__(
        self, data: JNPArray, num_data_points: int, PRNG_key: PRNGKey
    ) -> JNPArray:
        self.num_points.append(num_data_points)
        self.PRNG_key = PRNG_key
        return data
