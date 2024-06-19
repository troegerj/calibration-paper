# Standard library imports
from functools import partial
from typing import Callable, NamedTuple, Optional, Union

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray, JNPBool, JNPFloat

_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


class LineSearchState(NamedTuple):
    """State of a line search."""

    is_terminated: Union[bool, JNPBool]
    is_failed: Union[bool, JNPBool]
    iteration: int
    alpha_prev: Union[float, JNPFloat]
    phi_prev: Union[float, JNPFloat]
    grad_phi_prev: JNPArray
    grad_prev: JNPArray
    num_func_evals: int
    num_grad_evals: int
    alpha_star: Union[float, JNPFloat]
    phi_star: Union[float, JNPFloat]
    grad_star: JNPArray


class LineSearchResults(NamedTuple):
    """Results of a line search."""

    is_failed: Union[bool, JNPBool]
    step_length_k: Union[float, JNPFloat]
    func_kp1: Union[float, JNPFloat]
    grad_kp1: JNPArray
    num_iters: int
    num_func_evals: int
    num_grad_evals: int
    status: int


class ZoomState(NamedTuple):
    """State of the zoom stage."""

    is_terminated: Union[bool, JNPBool]
    is_failed: Union[bool, JNPBool]
    iteration: int
    alpha_lo: Union[float, JNPFloat]
    phi_lo: Union[float, JNPFloat]
    grad_phi_lo: JNPArray
    alpha_hi: Union[float, JNPFloat]
    phi_hi: Union[float, JNPFloat]
    grad_phi_hi: JNPArray
    alpha_rec: Union[float, JNPFloat]
    phi_rec: Union[float, JNPFloat]
    alpha_star: Union[float, JNPFloat]
    phi_star: Union[float, JNPFloat]
    grad_star: JNPArray
    num_func_evals: int
    num_grad_evals: int


def zoom(
    line_func: Callable[[Union[float, JNPFloat]], tuple[JNPFloat, JNPArray, JNPArray]],
    wolfe_cond_one: Callable[[Union[float, JNPFloat], Union[float, JNPFloat]], JNPBool],
    wolfe_cond_two: Callable[[JNPArray], JNPBool],
    alpha_lo: Union[float, JNPFloat],
    phi_lo: Union[float, JNPFloat],
    grad_phi_lo: JNPArray,
    grad_lo: JNPArray,
    alpha_hi: Union[float, JNPFloat],
    phi_hi: Union[float, JNPFloat],
    grad_phi_hi: JNPArray,
    max_iters: int,
    is_pass_through: Union[bool, JNPBool],
) -> ZoomState:
    state = ZoomState(
        is_terminated=False,
        is_failed=False,
        iteration=0,
        alpha_lo=alpha_lo,
        phi_lo=phi_lo,
        grad_phi_lo=grad_phi_lo,
        alpha_hi=alpha_hi,
        phi_hi=phi_hi,
        grad_phi_hi=grad_phi_hi,
        alpha_rec=(alpha_lo + alpha_hi) / 2.0,
        phi_rec=(phi_lo + phi_hi) / 2.0,
        # Use other initial values for alpha_star, phi_star and grad_star as in   jax.scipy. These values will be returned if zoom fails.
        alpha_star=alpha_lo,
        phi_star=phi_lo,
        grad_star=grad_lo,
        num_func_evals=0,
        num_grad_evals=0,
    )

    # As per jax.scipy.
    safeguard_factor_cubic = 0.2
    safeguard_factor_quad = 0.1

    def run_one_iteration(state):
        d_alpha = state.alpha_hi - state.alpha_lo
        alpha_min = jnp.minimum(state.alpha_lo, state.alpha_hi)
        alpha_max = jnp.maximum(state.alpha_lo, state.alpha_hi)
        safeguard_cubic = safeguard_factor_cubic * d_alpha
        safeguard_quad = safeguard_factor_quad * d_alpha

        # This check cause the line search to stop if two consecutive alphas are indistinguishable in finite-precision arithmetic. Since in case of a stop, the Wolfe conditions are not satisfied the optimization should stop too.
        precision_threshold = jnp.where((jnp.finfo(d_alpha).bits < 64), 1e-5, 1e-10)
        state._replace(is_failed=(d_alpha <= precision_threshold) | state.is_failed)

        alpha_j_cubic = _minimize_cubic(
            alpha_lo=state.alpha_lo,
            phi_lo=state.phi_lo,
            grad_phi_lo=state.grad_phi_lo,
            alpha_hi=state.alpha_hi,
            phi_hi=state.phi_hi,
            alpha_rec=state.alpha_rec,
            phi_rec=state.phi_rec,
        )
        use_cubic = (
            (state.iteration > 0)
            & (alpha_j_cubic > alpha_min + safeguard_cubic)
            & (alpha_j_cubic < alpha_max - safeguard_cubic)
        )
        alpha_j_quad = _minimize_quadratic(
            alpha_lo=state.alpha_lo,
            phi_lo=state.phi_lo,
            grad_phi_lo=state.grad_phi_lo,
            alpha_hi=state.alpha_hi,
            phi_hi=state.phi_hi,
        )
        use_quad = (
            (~use_cubic)
            & (alpha_j_quad > alpha_min + safeguard_quad)
            & (alpha_j_quad < alpha_max - safeguard_quad)
        )
        alpha_j_bisection = (state.alpha_hi + state.alpha_lo) / 2.0
        use_bisection = (~use_cubic) & (~use_quad)

        alpha_j = jnp.where(use_cubic, alpha_j_cubic, state.alpha_rec)
        alpha_j = jnp.where(use_quad, alpha_j_quad, alpha_j)
        alpha_j = jnp.where(use_bisection, alpha_j_bisection, alpha_j)

        phi_j, grad_phi_j, grad_j = line_func(alpha_j)
        # TODO: Check why this is necessary.
        # phi_j = phi_j.astype(state.phi_lo.dtype)
        # grad_phi_j = grad_phi_j.astype(state.grad_phi_lo.dtype)
        # grad_j = grad_j.astype(state.grad_star.dtype)
        state._replace(
            num_func_evals=state.num_func_evals + 1,
            num_grad_evals=state.num_grad_evals + 1,
        )

        # First Wolfe condition is violated or phi_j>phi_lo. alpha_lo remains unchanged.
        hi_to_j = ~wolfe_cond_one(alpha_j, phi_j) | (phi_j >= state.phi_lo)
        # In addition to the first (j_is_hi==False), the second Wolfe condition is also satidfied.
        star_to_j = (~hi_to_j) & wolfe_cond_two(grad_phi_j)
        # Only the first Wolfe condiion (suffiecient decrease condition) is satisfied.
        lo_to_j = (~hi_to_j) & (~star_to_j)
        # alpha_hi is chosen depending on grad_alpha_lo which is grad_alpha_j since j_is_lo==True.
        hi_to_old_lo = lo_to_j & (grad_phi_j * (state.alpha_hi - state.alpha_lo) >= 0.0)

        state = state._replace(
            **_binary_replace(
                hi_to_j,
                state._asdict(),
                dict(
                    alpha_hi=alpha_j,
                    phi_hi=phi_j,
                    grad_phi_hi=grad_phi_j,
                    alpha_rec=state.alpha_hi,
                    phi_rec=state.phi_hi,
                ),
            )
        )
        state = state._replace(
            is_terminated=star_to_j | state.is_terminated,
            **_binary_replace(
                star_to_j,
                state._asdict(),
                dict(
                    alpha_star=alpha_j,
                    phi_star=phi_j,
                    grad_star=grad_j,
                ),
            )
        )
        # Old lo values are needed if hi_to_old_lo=True.
        old_alpha_lo = state.alpha_lo
        old_phi_lo = state.phi_lo
        old_grad_phi_lo = state.grad_phi_lo
        state = state._replace(
            **_binary_replace(
                lo_to_j,
                state._asdict(),
                dict(
                    alpha_lo=alpha_j,
                    phi_lo=phi_j,
                    grad_phi_lo=grad_phi_j,
                    alpha_rec=state.alpha_lo,
                    phi_rec=state.phi_lo,
                ),
            )
        )
        # Instead of old lo values you can also use rec values. They are identic.
        state = state._replace(
            **_binary_replace(
                hi_to_old_lo,
                state._asdict(),
                dict(
                    alpha_hi=old_alpha_lo,
                    phi_hi=old_phi_lo,
                    grad_phi_hi=old_grad_phi_lo,
                    alpha_rec=state.alpha_hi,
                    phi_rec=state.phi_hi,
                ),
            )
        )

        state = state._replace(iteration=state.iteration + 1)
        return state

    cond_func = (
        lambda state: (~is_pass_through)
        & (~state.is_terminated)
        & (~state.is_failed)
        & (state.iteration < max_iters)
    )
    state = jax.lax.while_loop(cond_func, run_one_iteration, state)
    # Reaching the maximum zoom iteration without terminating causes the zoom to fail.
    state._replace(
        is_failed=((state.iteration >= max_iters) & ~state.is_terminated)
        | state.is_failed
    )
    return state


def line_search(
    params_0: JNPArray,
    loss_func: Callable[[JNPArray], JNPFloat],
    search_direction: JNPArray,
    loss_0: JNPFloat,
    grad_0: JNPArray,
    c1: float,
    c2: float,
    max_bracketing_iters: int,
    max_zoom_iters: int,
) -> LineSearchResults:
    """Inexact line search satisfying the stron Wolf conditions.

    Implementation of Algorithm 3.6 from Wright and Nocedal, 'Numercial Optimization', 2006, 686 pages.
    """

    def line_func(
        step_length: Union[float, JNPFloat]
    ) -> tuple[JNPFloat, JNPArray, JNPArray]:
        phi, grad = jax.value_and_grad(loss_func)(
            params_0 + step_length * search_direction
        )
        # The gradient of the loss function does not necessariliy points in the search direction. It can be calculated from the dot product of the gradient and the search direction.
        # TODO: Check if the dot product can be a complex vector.
        grad_phi = _dot(grad, search_direction)
        return phi, grad_phi, grad

    def wolfe_cond_one(alpha_i: Union[float, JNPFloat], phi_i: JNPFloat) -> JNPBool:
        return phi_i <= phi_0 + c1 * alpha_i * grad_phi_0

    def wolfe_cond_two(grad_phi_i: JNPArray) -> JNPBool:
        return jnp.abs(grad_phi_i) <= -c2 * grad_phi_0

    is_loss_and_grad_0_given = ~(loss_0 is None or grad_0 is None)
    if not is_loss_and_grad_0_given:
        phi_0, grad_phi_0, grad_0 = line_func(0.0)
    else:
        phi_0 = loss_0
        grad_phi_0 = _dot(grad_0, search_direction)

    alpha_init = 1.0

    # Line search starts at alpha_0 = 0.0 with the corresponding parameters, loss and gradient params_0, loss_0 and grad_0, respectively.
    state = LineSearchState(
        is_terminated=False,
        is_failed=False,
        iteration=1,  # Starts with 1 as per Wright and Nocedal.
        alpha_prev=0.0,
        phi_prev=phi_0,
        grad_phi_prev=grad_phi_0,
        grad_prev=grad_0,
        num_func_evals=1 if (not is_loss_and_grad_0_given) else 0,
        num_grad_evals=1 if (not is_loss_and_grad_0_given) else 0,
        alpha_star=0.0,
        phi_star=phi_0,
        grad_star=grad_0,
    )

    def run_one_iteration(state):
        # In this version no alpha_max is used. alpha is just doubled in each iteration as in jax.scipy. However, the maximum number of iterations is limited to max_iters.
        # Unlike original algorithm we do our next choice at the start of this loop.

        alpha_i = jnp.where(state.iteration == 1, alpha_init, 2 * state.alpha_prev)

        phi_i, grad_phi_i, grad_i = line_func(alpha_i)
        state = state._replace(
            num_func_evals=state.num_func_evals + 1,
            num_grad_evals=state.num_grad_evals + 1,
        )

        is_zoom_1 = ~wolfe_cond_one(alpha_i, phi_i) | (
            (phi_i >= state.phi_prev) & (state.iteration > 1)
        )
        is_star = wolfe_cond_two(grad_phi_i) & (~is_zoom_1)
        is_zoom_2 = (grad_phi_i >= 0.0) & (~is_zoom_1) & (~is_star)

        # alpha_lo is among all step length generated so far and sattisfying the sufficient decrease condition the one resulting in the smallest function value.
        # is_zoom_1: If the first Wolfe condition is not met, then the previous step length (alpha_prev) is the step length with the smallest function value. This must be true because of the second codition. According to this condition, alpha_i was never greater than alpha_prev.
        # is_zoom_2: Since the first Wolfe condition is fullfilled and alpha_i is less than alpha_prev, alph_i must be the step length with the smallest function value.

        zoom_results_1 = zoom(
            line_func=line_func,
            wolfe_cond_one=wolfe_cond_one,
            wolfe_cond_two=wolfe_cond_two,
            alpha_lo=state.alpha_prev,
            phi_lo=state.phi_prev,
            grad_phi_lo=state.grad_phi_prev,
            grad_lo=state.grad_prev,
            alpha_hi=alpha_i,
            phi_hi=phi_i,
            grad_phi_hi=grad_phi_i,
            max_iters=max_zoom_iters,
            is_pass_through=(~is_zoom_1),
        )
        state = state._replace(
            num_func_evals=state.num_func_evals + zoom_results_1.num_func_evals,
            num_grad_evals=state.num_grad_evals + zoom_results_1.num_grad_evals,
        )

        zoom_results_2 = zoom(
            line_func=line_func,
            wolfe_cond_one=wolfe_cond_one,
            wolfe_cond_two=wolfe_cond_two,
            alpha_lo=alpha_i,
            phi_lo=phi_i,
            grad_phi_lo=grad_phi_i,
            grad_lo=grad_i,
            alpha_hi=state.alpha_prev,
            phi_hi=state.phi_prev,
            grad_phi_hi=state.grad_phi_prev,
            max_iters=max_zoom_iters,
            is_pass_through=(~is_zoom_2),
        )
        state = state._replace(
            num_func_evals=state.num_func_evals + zoom_results_2.num_func_evals,
            num_grad_evals=state.num_grad_evals + zoom_results_2.num_grad_evals,
        )

        state = state._replace(
            is_terminated=is_zoom_1 | state.is_terminated,
            is_failed=(is_zoom_1 & zoom_results_1.is_failed) | state.is_failed,
            **_binary_replace(
                is_zoom_1,
                state._asdict(),
                zoom_results_1._asdict(),
                keys=("alpha_star", "phi_star", "grad_star"),
            )
        )
        state = state._replace(
            is_terminated=is_star | state.is_terminated,
            **_binary_replace(
                is_star,
                state._asdict(),
                dict(
                    alpha_star=alpha_i,
                    phi_star=phi_i,
                    grad_star=grad_i,
                ),
            )
        )
        state = state._replace(
            is_terminated=is_zoom_2 | state.is_terminated,
            is_failed=(is_zoom_2 & zoom_results_2.is_failed) | state.is_failed,
            **_binary_replace(
                is_zoom_2,
                state._asdict(),
                zoom_results_2._asdict(),
                keys=("alpha_star", "phi_star", "grad_star"),
            )
        )
        state = state._replace(
            iteration=state.iteration + 1,
            alpha_prev=alpha_i,
            phi_prev=phi_i,
            grad_phi_prev=grad_phi_i,
            grad_prev=grad_i,
        )
        return state

    cond_func = (
        lambda state: (~state.is_terminated)
        & (~state.is_failed)
        # <= as iteration starts from 1.
        & (state.iteration <= max_bracketing_iters)
    )
    state = jax.lax.while_loop(cond_func, run_one_iteration, state)

    is_zoom_failed = (
        state.is_failed
    )  # Up to here the failure can only be caused by zoom.

    # Reaching the maximum bracketing iterations without terminating causes the search to fail.
    state._replace(
        is_failed=((state.iteration > max_bracketing_iters) & ~state.is_terminated)
        | state.is_failed
    )

    status = jnp.where(
        is_zoom_failed,
        jnp.array(3),  # zoom failed
        jnp.where(
            state.iteration > max_bracketing_iters,
            jnp.array(4),  # maximum line bracketing iterations reached
            jnp.array(1),  # pass (optimization in progress)
        ),
    )
    alpha_k = state.alpha_star
    alpha_k = jnp.where(
        (jnp.finfo(alpha_k).bits != 64) & (jnp.abs(alpha_k) < 1e-8),
        jnp.sign(alpha_k) * 1e-8,
        alpha_k,
    )
    results = LineSearchResults(
        is_failed=state.is_failed & (~state.is_terminated),
        step_length_k=alpha_k,
        func_kp1=state.phi_star,
        grad_kp1=state.grad_star,
        num_iters=state.iteration - 1,  # Because iteration starts at 1.
        num_func_evals=state.num_func_evals,
        num_grad_evals=state.num_grad_evals,
        status=status,
    )
    return results


def _minimize_quadratic(alpha_lo, phi_lo, grad_phi_lo, alpha_hi, phi_hi):
    """
    Calculates the minimizer of an intepolated quadratic function.

    The quadratic function has the function values phi_lo and phi_hi at alpha_lo and alpha_hi and the first derivative d_phi_lo at alpha_lo.
    """
    d_alpha = alpha_lo - alpha_hi
    a = ((phi_lo - phi_hi - grad_phi_lo * d_alpha)) / (-(d_alpha**2))
    b = grad_phi_lo - 2 * a * alpha_lo
    x_min = -(b / 2 * a)
    return x_min


def _minimize_cubic(
    alpha_lo, phi_lo, grad_phi_lo, alpha_hi, phi_hi, alpha_rec, phi_rec
):
    """
    Calculates the minimizer of an intepolated cubic function.

    The cubicic function has the function values phi_lo, phi_hi, phi_rec at alpha_lo, alpha_hi and alpha_rec and the first derivative d_phi_lo at alpha_lo. To simplify the calculation of the minimizer, the support points are shifted by -alpha_lo. After the minimizer of the cubic function is calculated, it is reshifted by alpha_lo.
    """
    d_alpha_rec = alpha_rec - alpha_lo
    d_alpha_hi = alpha_hi - alpha_lo
    # In jax.scipy: d_alpha_hi - d_alpha_rec
    denominator = d_alpha_hi**2 * d_alpha_rec**2 * (d_alpha_rec - d_alpha_hi)
    arr_alphas = jnp.array(
        [
            [d_alpha_hi**2, -(d_alpha_rec**2)],
            [-(d_alpha_hi**3), d_alpha_rec**3],
        ]
    )
    arr_phis = jnp.array(
        [
            phi_rec - phi_lo - grad_phi_lo * d_alpha_rec,
            phi_hi - phi_lo - grad_phi_lo * d_alpha_hi,
        ]
    )
    a, b = _dot(arr_alphas, arr_phis) / denominator
    radicand = b**2 - 3.0 * a * grad_phi_lo
    alpha_min = alpha_lo + (-b + jnp.sqrt(radicand)) / (3.0 * a)
    return alpha_min


def _binary_replace(
    should_replace: bool,
    original_dict: dict,
    new_dict: dict,
    keys: Optional[tuple[str, ...]] = None,
):
    keys = keys or tuple(new_dict.keys())
    output_dict = dict()
    for key in keys:
        output_dict[key] = jnp.where(should_replace, new_dict[key], original_dict[key])
    return output_dict
