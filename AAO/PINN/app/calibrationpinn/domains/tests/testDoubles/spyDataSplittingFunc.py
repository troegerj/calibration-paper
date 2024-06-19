# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.domains.splitDataFunc import DataSplittingFuncProtocol
from calibrationpinn.typeAliases import JNPArray, PRNGKey


class SpyDataSplittingFunc(DataSplittingFuncProtocol):
    def __init__(self) -> None:
        self.proportion_train_data: float
        self.PRNG_key: PRNGKey

    def __call__(
        self, data: JNPArray, proportion_train_data: float, PRNG_key: PRNGKey
    ) -> tuple[JNPArray, JNPArray]:
        self.proportion_train_data = proportion_train_data
        self.PRNG_key = PRNG_key
        return data, data
