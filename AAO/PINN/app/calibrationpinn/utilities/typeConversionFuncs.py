# Standard library imports
from typing import Protocol

# Third-party imports
import jax.numpy as jnp
import numpy as np

# Local library imports
from calibrationpinn.typeAliases import NPArray, JNPArray


class NumpyToJAXNumpyProtocol(Protocol):
    def __call__(self, np_array: NPArray) -> JNPArray:
        """
        Converts umpy array to jax.numpy array.
        """


def numpy_to_jax_numpy(np_array: NPArray) -> JNPArray:
    return jnp.array(np_array)


def jax_numpy_to_numpy(jnp_array: JNPArray) -> NPArray:
    return np.array(jnp_array)
