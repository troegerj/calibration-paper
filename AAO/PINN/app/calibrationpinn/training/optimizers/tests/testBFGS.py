# Standard library imports
from functools import partial
import types
from typing import Any, Callable, Protocol, TypeAlias, Union
import unittest

# Third-party imports
import jax
import jax.numpy as jnp
import optax

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.training.optimizers.BFGS import BFGS, BFGSOptimizerState
from calibrationpinn.typeAliases import (
    JNPArray,
    JNPFloat,
    NPFloat,
    OptaxGradientTransformation,
)

FloatLike: TypeAlias = Union[NPFloat, JNPFloat]


def rosenbrock_func(np_pkg: types.ModuleType) -> Callable[..., FloatLike]:
    def _rosenbrock_func(x: Any) -> FloatLike:
        a = 1.0
        b = 5.0
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    return _rosenbrock_func


def ackley_func(np_pkg: types.ModuleType) -> Callable[..., FloatLike]:
    def _ackley_func(x: Any) -> FloatLike:
        a = 20.0
        b = 0.2
        c = 2 * np_pkg.pi
        length = x.shape[0]
        return (
            -a * np_pkg.exp(-b * np_pkg.sqrt(np_pkg.sum(x**2) / length))
            - np_pkg.exp(np_pkg.sum(np_pkg.cos(c * x) / length))
            + a
            + np_pkg.exp(1.0)
        )

    return _ackley_func


class TestBFGSWithRosenbrock(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._loss_func = jax.jit(rosenbrock_func(jnp))
        self._max_iters = 100
        self._sut = BFGS()
        self._init_x = jnp.array([-1.0, -1.0])
        self._init_optimizer_state = self._sut.init(
            params=self._init_x, loss_func=self._loss_func, func_args=()
        )

    def test_bfgs(self) -> None:
        """Test that the BFGS algorithm finds the global minimum."""

        actual, _ = minimize(
            test_case=self, x=self._init_x, optimizer_state=self._init_optimizer_state
        )
        expected = jnp.array([1.0, 1.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual, atol=1e-8)


# class TestBFGSWithAckley(unittest.TestCase):
#     def __init__(self, methodName: str) -> None:
#         super().__init__(methodName=methodName)

#     def setUp(self) -> None:
#         self._loss_func = jax.jit(ackley_func(jnp))
#         self._max_iters = 100
#         self._sut = BFGS()
#         self._init_x = jnp.array([1.0, 1.0])
#         self._init_optimizer_state = self._sut.init(
#             params=self._init_x, loss_func=self._loss_func, func_args=()
#         )

#     def test_bfgs(self):
#         """Test that the BFGS algorithm finds the global minimum."""

#         actual, _ = minimize(
#             test_case=self, x=self._init_x, optimizer_state=self._init_optimizer_state
#         )
#         expected = jnp.array([0.0, 0.0])
#         print(actual)
#         assert_equal_arrays(self=self, expected=expected, actual=actual, atol=1e-8)


class TestCaseProtocol(Protocol):
    _loss_func: Callable[..., FloatLike]
    _max_iters: int
    _sut: OptaxGradientTransformation


@partial(jax.jit, static_argnums=(0,))
def minimize(
    test_case: TestCaseProtocol, x: JNPArray, optimizer_state: BFGSOptimizerState
) -> tuple[JNPArray, BFGSOptimizerState]:
    for _ in range(test_case._max_iters):
        updates, optimizer_state = test_case._sut.update(
            params=x,
            loss_func=test_case._loss_func,
            func_args=(),
            optimizer_state=optimizer_state,
        )
        x = optax.apply_updates(x, updates)
    return x, optimizer_state
