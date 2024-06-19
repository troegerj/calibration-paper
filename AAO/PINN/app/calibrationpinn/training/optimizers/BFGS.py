# Standard library imports
from functools import partial
from typing import Any, Callable, NamedTuple, TypeAlias, Union

# Third-party imports
import jax
import jax.numpy as jnp
from optax._src import base

# Local library imports
from calibrationpinn.training.optimizers.lineSearch import line_search
from calibrationpinn.utilities import (
    parameters_to_array,
    array_to_parameters,
    ParametersDefinition,
)
from calibrationpinn.typeAliases import JNPArray, JNPBool, JNPFloat, JNPPyTree


BoolLike: TypeAlias = Union[bool, JNPBool]
LossFunc: TypeAlias = Callable[[JNPArray], JNPFloat]

_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


class BFGSOptimizerState(NamedTuple):
    """Optimizer state for the BFGS algorithm."""

    is_terminated: BoolLike
    is_converged: BoolLike
    params_k: JNPPyTree
    func_k: JNPFloat
    grad_k: JNPArray
    inv_hessian_k: JNPArray
    iteration: int
    num_func_evals: int
    num_grad_evals: int
    num_inv_hessian_evals: int
    status: int


def BFGS(
    c1: float = 1e-4,
    c2: float = 0.9,
    bracket_maxiter: int = 20,
    zoom_maxiter: int = 30,
    gtol: float = 1e-5,
) -> base.GradientTransformation:
    def init_func(
        params: JNPPyTree, loss_func: Callable[..., JNPFloat], func_args: tuple[Any]
    ) -> BFGSOptimizerState:
        params_0, params_def = parameters_to_array(params)
        _loss_func = _accept_params_array(loss_func, params_def)
        func_0, grad_0 = jax.value_and_grad(_loss_func, 0)(params_0, func_args)
        inv_hessian_0 = jnp.eye(params_0.shape[0], dtype=params_0.dtype)
        is_converged = is_terminated = _check_if_converged(grad_0)
        status = jnp.where(is_converged, 2, 0)
        return BFGSOptimizerState(
            is_terminated=is_terminated,
            is_converged=is_converged,
            params_k=params,
            grad_k=grad_0,
            func_k=func_0,
            inv_hessian_k=inv_hessian_0,
            iteration=0,
            num_func_evals=1,
            num_grad_evals=1,
            num_inv_hessian_evals=0,
            status=status,
        )

    def update_func(
        params: JNPPyTree,
        loss_func: LossFunc,
        func_args: tuple[Any],
        optimizer_state: BFGSOptimizerState,
    ) -> tuple[JNPPyTree, BFGSOptimizerState]:
        """Update function of the BFGS algorithm."""

        params_k, params_def = parameters_to_array(params)

        _loss_func = _make_func_args_implicit(
            _accept_params_array(loss_func, params_def), func_args
        )
        dim_params = params_k.shape[0]

        # Line search
        func_k = optimizer_state.func_k
        grad_k = optimizer_state.grad_k
        inv_hessian_k = optimizer_state.inv_hessian_k
        search_direction_k = -_dot(inv_hessian_k, grad_k)
        line_search_results_k = line_search(
            params_0=params_k,
            func_0=func_k,
            grad_0=grad_k,
            loss_func=_loss_func,
            search_direction=search_direction_k,
            c1=c1,
            c2=c2,
            bracket_maxiters=bracket_maxiter,
            zoom_maxiters=zoom_maxiter,
        )

        # Parameters update
        step_length_k = line_search_results_k.step_length_k
        param_update_k = step_length_k * search_direction_k
        params_kp1 = params_k + param_update_k
        func_kp1 = line_search_results_k.func_kp1
        grad_kp1 = line_search_results_k.grad_kp1

        # Gradient update
        grad_update_k = grad_kp1 - grad_k

        # Inverse hessian update
        rho_k = jnp.reciprocal(_dot(grad_update_k, param_update_k))
        # params_update_k * grad_update_k^T = (grad_update_k * params_update_k^T)^T
        # For this reason, the third multiplicand in inv_hessian_kp1 is transposed.
        interim_result_1 = jnp.expand_dims(param_update_k, 1) * jnp.expand_dims(
            grad_update_k, 0
        )
        interim_result_2 = (
            jnp.eye(dim_params, dtype=params_k.dtype) - rho_k * interim_result_1
        )
        # Subscripts are "ij,jk,kl" and not "ij,jk,kl" as the third operand is transposed.
        inv_hessian_kp1 = _einsum(
            "ij,jk,lk", interim_result_2, inv_hessian_k, interim_result_2
        ) + rho_k * jnp.expand_dims(param_update_k, 1) * jnp.expand_dims(
            param_update_k, 0
        )
        inv_hessian_kp1 = jnp.where(jnp.isfinite(rho_k), inv_hessian_kp1, inv_hessian_k)

        # Update state
        num_func_evals = (
            optimizer_state.num_func_evals + line_search_results_k.num_func_evals
        )
        num_grad_evals = (
            optimizer_state.num_grad_evals + line_search_results_k.num_grad_evals
        )
        num_inv_hessian_evals = optimizer_state.num_inv_hessian_evals + 1
        params_kp1_pytree = array_to_parameters(params_kp1, params_def)

        is_converged = _check_if_converged(grad_kp1)
        is_terminated = is_converged | line_search_results_k.is_failed

        status = jnp.where(
            is_converged,
            2,  # optimization converged
            jnp.where(
                line_search_results_k.is_failed,
                line_search_results_k.status,  # reason for line search failure
                1,  # optimization in progress
            ),
        )

        # Optimizer state for next itaeration.
        optimizer_state = BFGSOptimizerState(
            is_terminated=is_terminated,
            is_converged=is_converged,
            params_k=params_kp1_pytree,
            func_k=func_kp1,
            grad_k=grad_kp1,
            inv_hessian_k=inv_hessian_kp1,
            iteration=optimizer_state.iteration + 1,
            num_func_evals=num_func_evals,
            num_grad_evals=num_grad_evals,
            num_inv_hessian_evals=num_inv_hessian_evals,
            status=status,
        )

        param_update_k_pytree = array_to_parameters(param_update_k, params_def)
        return param_update_k_pytree, optimizer_state

    def _accept_params_array(
        loss_func: Callable[..., JNPFloat], params_def: ParametersDefinition
    ) -> Callable[..., JNPFloat]:
        def _loss_func(params: JNPArray, func_args: tuple[Any]) -> JNPFloat:
            params_pytree = array_to_parameters(params, params_def)
            return loss_func(params_pytree, *func_args)

        return _loss_func

    def _make_func_args_implicit(
        loss_func: Callable[..., JNPFloat], func_args: tuple[Any]
    ) -> Callable[[JNPArray], JNPFloat]:
        def _loss_func(params: JNPArray) -> JNPFloat:
            return loss_func(params, func_args)

        return _loss_func

    def _check_if_converged(grad: JNPArray) -> JNPBool:
        return jnp.linalg.norm(grad, ord=jnp.inf) < gtol

    return base.GradientTransformation(init_func, update_func)
