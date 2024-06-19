# Standard library imports
from typing import NamedTuple, TypeAlias, Union

# Third-party imports
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

# Local library imports
from calibrationpinn.errors import TypeNotSupportedError
from calibrationpinn.typeAliases import JNPArray, JNPFloat, JNPPyTree, JNPPyTreeDef


# Mathematical operations for pytrees
def sum_pytrees(pytrees: JNPPyTree) -> JNPPyTree:
    sum_pytrees = pytrees[0]
    for pytree in pytrees[1:]:
        sum_pytrees = tree_map(lambda x, y: x + y, sum_pytrees, pytree)
    return sum_pytrees


def mul_pytrees(factor: float, pytree: JNPPyTree) -> JNPPyTree:
    pytree_flat, tree_def = tree_flatten(pytree)
    multiplied_pytree_flat = [factor * x for x in pytree_flat]
    return tree_unflatten(tree_def, multiplied_pytree_flat)


# Functions to flatten and unflatten trainable parameters
Shapes: TypeAlias = Union[list[tuple[int]], list[tuple[int, ...]]]
Sizes: TypeAlias = list[int]
Leave: TypeAlias = Union[JNPArray, JNPFloat]


class ParametersDefinition(NamedTuple):
    tree_definition: JNPPyTreeDef
    shapes: Shapes
    sizes: Sizes


def parameters_to_array(parameters: JNPPyTree) -> tuple[JNPArray, ParametersDefinition]:
    leaves, tree_definition = tree_flatten(parameters)
    shapes: Shapes = []
    sizes: Sizes = []
    leaves_flatten: list[JNPArray] = []

    for leave in leaves:
        if isinstance(leave, jnp.ndarray):
            shapes.append(leave.shape)
            leave_flatten = jnp.ravel(leave, order="C")
            sizes.append(leave_flatten.size)
        else:
            raise TypeNotSupportedError(type(leave))
        leaves_flatten.append(leave_flatten)

    parameters_array = jnp.concatenate(tuple(leaves_flatten))
    parameters_definition = ParametersDefinition(tree_definition, shapes, sizes)
    return parameters_array, parameters_definition


def array_to_parameters(
    parameters_array: JNPArray, parameters_definition: ParametersDefinition
) -> JNPPyTree:
    tree_definition = parameters_definition.tree_definition
    shapes = parameters_definition.shapes
    sizes = parameters_definition.sizes
    leaves: list[Leave] = []
    parameter_ptr: int = 0

    for index_shape, shape in enumerate(shapes):
        size = sizes[index_shape]
        if isinstance(shape, tuple):
            parameters_leave = parameters_array[parameter_ptr : parameter_ptr + size]
            leave = parameters_leave.reshape(shape)
            leaves.append(leave)
        else:
            raise TypeNotSupportedError(type(shape))
        parameter_ptr = parameter_ptr + size

    return tree_unflatten(tree_definition, leaves)
