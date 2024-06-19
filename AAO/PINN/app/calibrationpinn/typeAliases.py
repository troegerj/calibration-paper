# Standard library imports
from typing import Any, Union, TypeAlias, NamedTuple

# Third-party imports
import haiku
import haiku._src.typing as hkt
import jax.numpy as jnp
from jax._src import prng
from jax._src.lib import pytree
from matplotlib import figure, axes
import numpy as np
import numpy.typing as npt
from optax._src import base as optax_base
import pandas as pd

# Local library imports


# Python
PYNumbers: TypeAlias = Union[int, float, complex]

# Numpy
NPArray: TypeAlias = Union[
    npt.NDArray[np.int16],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]
NPFloat: TypeAlias = Union[np.float32, np.float64]
NPFloatTuple: TypeAlias = Union[tuple[np.float32, ...], tuple[np.float64, ...]]

# JAX
JNPArray: TypeAlias = jnp.ndarray
JNPFloat: TypeAlias = Union[jnp.float32, jnp.float64]
JNPBool: TypeAlias = jnp.bool_
JNPPyTree: TypeAlias = Union[NamedTuple, list[Any], tuple[Any], dict[Any, Any]]
JNPPyTreeDef: TypeAlias = Any
PRNGKey: TypeAlias = prng.PRNGKeyArray

# JAX and Numpy
Array: TypeAlias = Union[NPArray, JNPArray]

# Haiku
HKInitializer: TypeAlias = hkt.Initializer
HKTransformed: TypeAlias = haiku.Transformed
HKModule: TypeAlias = hkt.Module

# Optax
OptaxGradientTransformation = optax_base.GradientTransformation

# Pandas
PDDataFrame: TypeAlias = pd.DataFrame
PDDataType: TypeAlias = npt.DTypeLike
PDSeries: TypeAlias = pd.Series

# Matplotlib
PLTFigure: TypeAlias = figure.Figure
PLTAxes: TypeAlias = axes.Axes
