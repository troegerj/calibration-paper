# Standard library imports
from typing import Protocol

# Third-party imports
import jax

# Local library imports
from calibrationpinn.typeAliases import JNPArray, PRNGKey


class DataSelectionFuncProtocol(Protocol):
    def __call__(
        self, data: JNPArray, num_data_points: int, PRNG_key: PRNGKey
    ) -> JNPArray:
        """
        Selects the specified number of data points from the passed data.
        """


def select_data_randomly(
    data: JNPArray, num_data_points: int, PRNG_key: PRNGKey
) -> JNPArray:
    shuffled_data = jax.random.permutation(PRNG_key, data)
    return shuffled_data[:num_data_points]


def select_data_sequentially(
    data: JNPArray, num_data_points: int, PRNG_key: PRNGKey
) -> JNPArray:
    return data[:num_data_points]
