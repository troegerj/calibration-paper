# Standard library imports

# Third-party imports
from typing import Callable
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray, JNPFloat


def mean_squared_error(y_true: JNPArray, y_pred: JNPArray) -> JNPArray:
    return jnp.mean(jnp.square(y_pred - y_true))


def relative_mean_squared_error(
    characteristic_length: JNPFloat,
) -> Callable[[JNPArray, JNPArray], JNPArray]:
    def relative_mean_squared_error_inner(
        y_true: JNPArray, y_pred: JNPArray
    ) -> JNPArray:
        return jnp.mean(jnp.square((y_pred - y_true) / characteristic_length))

    return relative_mean_squared_error_inner


def mean_absolute_error(y_true: JNPArray, y_pred: JNPArray) -> JNPArray:
    return jnp.mean(jnp.abs(y_pred - y_true))


def relative_error(y_true: JNPArray, y_pred: JNPArray) -> JNPArray:
    return jnp.mean((y_pred - y_true) / y_true)


def l2_norm(y_true: JNPArray, y_pred: JNPArray) -> JNPArray:
    return jnp.sqrt(jnp.sum(jnp.square(y_pred - y_true)))


def relative_l2_norm(y_true: JNPArray, y_pred: JNPArray) -> JNPArray:
    def l2_norm(array: JNPArray) -> JNPArray:
        return jnp.sqrt(jnp.sum(jnp.square(array)))

    return jnp.divide(l2_norm(y_pred - y_true), l2_norm(y_true))
