# Standard library imports
import math
from typing import Protocol

# Third-party import
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray, PRNGKey


class DataSplittingFuncProtocol(Protocol):
    def __call__(
        self, data: JNPArray, proportion_train_data: float, PRNG_key: PRNGKey
    ) -> tuple[JNPArray, JNPArray]:
        """
        Splits data into training and validation data sets according to the passed proportion.
        """


def split_in_train_and_valid_data(
    data: JNPArray, proportion_train_data: float, PRNG_key: PRNGKey
) -> tuple[JNPArray, JNPArray]:
    if abs(proportion_train_data - 1.0) < 1e-6:
        return _use_for_train_and_valid_data(data)
    else:
        return _split_in_train_and_valid_data(data, proportion_train_data, PRNG_key)


def _use_for_train_and_valid_data(data: JNPArray) -> tuple[JNPArray, JNPArray]:
    train_data = valid_data = data
    return train_data, valid_data


def _split_in_train_and_valid_data(
    data: JNPArray, proportion_train_data: float, PRNG_key: PRNGKey
) -> tuple[JNPArray, JNPArray]:
    num_data = data.shape[0]
    num_train_data = int(math.floor(proportion_train_data * num_data))
    shuffled_data = jax.random.permutation(PRNG_key, data)
    train_data, valid_data = jnp.split(shuffled_data, [num_train_data])
    return train_data, valid_data
