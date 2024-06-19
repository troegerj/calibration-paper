# Standard library imports
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, TypeAlias, Union

# Third-party imports
import jax
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray, JNPBool, JNPFloat


LossFunc: TypeAlias = Callable[[JNPArray], JNPFloat]
LineFunc: TypeAlias = Callable[
    [Union[float, JNPFloat]], tuple[JNPFloat, JNPArray, JNPArray]
]
WolfeCondOne: TypeAlias = Callable[
    [Union[float, JNPFloat], Union[float, JNPFloat]], JNPBool
]
WolfeCondTwo: TypeAlias = Callable[[JNPArray], JNPBool]
BoolLike: TypeAlias = Union[bool, JNPBool]
FloatLike: TypeAlias = Union[float, JNPFloat]

_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)


class LineSearchState(NamedTuple):
    """State of a line search."""

    is_terminated: BoolLike
    is_star_found: BoolLike
    is_failed: BoolLike
    alpha_star: FloatLike
    phi_star: FloatLike
    grad_star: JNPArray
    num_func_evals: int
    num_grad_evals: int


class LineSearchResults(NamedTuple):
    """Results of a line search."""

    is_failed: BoolLike
    step_length_k: FloatLike
    func_kp1: FloatLike
    grad_kp1: JNPArray
    num_func_evals: int
    num_grad_evals: int
    status: int


def line_search(
    params_0: JNPArray,
    func_0: FloatLike,
    grad_0: JNPArray,
    loss_func: LossFunc,
    search_direction: JNPArray,
    c1: float,
    c2: float,
    bracket_maxiters: int,
    zoom_maxiters: int,
) -> LineSearchResults:
    """Inexact line search satisfying the strong Wolf conditions.

    Implementation of Algorithm 3.6 from J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. New York: Springer, 2006.
    """

    def line_func(alpha: FloatLike) -> tuple[JNPFloat, JNPFloat, JNPArray]:
        phi, grad = jax.value_and_grad(loss_func)(params_0 + alpha * search_direction)
        # The gradient of the loss function does not necessariliy points in the search direction. The component of the gradient in search direction is calculated from the dot product of gradient and search direction.
        grad_phi = _dot(grad, search_direction)
        return phi, grad_phi, grad

    def wolfe_cond_one(alpha_i: FloatLike, phi_i: FloatLike) -> JNPBool:
        return phi_i <= phi_0 + c1 * alpha_i * grad_phi_0

    def wolfe_cond_two(grad_phi_i: FloatLike) -> JNPBool:
        return jnp.abs(grad_phi_i) <= -c2 * grad_phi_0

    is_func_0_and_grad_0 = ~((func_0 is None) | (grad_0 is None))
    if not is_func_0_and_grad_0:
        phi_0, grad_phi_0, grad_0 = line_func(0.0)
    else:
        phi_0 = func_0
        grad_phi_0 = _dot(grad_0, search_direction)

    alpha_init = 1.0

    state = LineSearchState(
        is_terminated=False,
        is_star_found=False,
        is_failed=False,
        alpha_star=0.0,
        phi_star=phi_0,
        grad_star=grad_0,
        num_func_evals=1 if (not is_func_0_and_grad_0) else 0,
        num_grad_evals=1 if (not is_func_0_and_grad_0) else 0,
    )

    bracket_results = _bracket(
        line_func=line_func,
        wolfe_cond_one=wolfe_cond_one,
        wolfe_cond_two=wolfe_cond_two,
        alpha_init=alpha_init,
        phi_0=phi_0,
        grad_phi_0=grad_phi_0,
        grad_0=grad_0,
        max_iters=bracket_maxiters,
    )
    state = state._replace(
        is_star_found=bracket_results.is_star_found | state.is_star_found,
        is_failed=bracket_results.is_failed | state.is_failed,
        num_func_evals=state.num_func_evals + bracket_results.num_func_evals,
        num_grad_evals=state.num_grad_evals + bracket_results.num_grad_evals,
    )
    state = state._replace(
        **_binary_replace(
            bracket_results.is_star_found,
            state._asdict(),
            dict(
                alpha_star=bracket_results.alpha_star,
                phi_star=bracket_results.phi_star,
                grad_star=bracket_results.grad_star,
            ),
        )
    )
    state = state._replace(is_terminated=state.is_star_found | state.is_failed)

    zoom_results = _zoom(
        line_func=line_func,
        wolfe_cond_one=wolfe_cond_one,
        wolfe_cond_two=wolfe_cond_two,
        alpha_lo=bracket_results.alpha_lo,
        phi_lo=bracket_results.phi_lo,
        grad_phi_lo=bracket_results.grad_phi_lo,
        grad_lo=bracket_results.grad_lo,
        alpha_hi=bracket_results.alpha_hi,
        phi_hi=bracket_results.phi_hi,
        grad_phi_hi=bracket_results.grad_phi_hi,
        max_iters=zoom_maxiters,
        is_pass_through=state.is_terminated,
    )
    state = state._replace(
        is_star_found=zoom_results.is_star_found | state.is_star_found,
        is_failed=zoom_results.is_failed | state.is_failed,
        num_func_evals=state.num_func_evals + zoom_results.num_func_evals,
        num_grad_evals=state.num_grad_evals + zoom_results.num_grad_evals,
    )
    state = state._replace(
        **_binary_replace(
            zoom_results.is_star_found,
            state._asdict(),
            dict(
                alpha_star=zoom_results.alpha_star,
                phi_star=zoom_results.phi_star,
                grad_star=zoom_results.grad_star,
            ),
        )
    )
    state = state._replace(is_terminated=state.is_star_found | state.is_failed)

    status = jnp.where(
        state.is_failed, -1, 0
    )  # -1: If the error does not occur in zoom or bracket stage.
    status = jnp.where(
        bracket_results.is_failed,
        jnp.where(bracket_results.is_max_iters_fail, -10, -11),
        status,
    )  # -11: The error occurs in the bracket stage, but for a reason other than reaching the maximum number of iterations.
    status = jnp.where(
        zoom_results.is_failed,
        jnp.where(zoom_results.is_max_iters_fail, -20, -21),
        status,
    )  # -21: The error occurs in the zoom stage, but for a reason other than reaching the maximum number of iterations.

    return LineSearchResults(
        is_failed=state.is_failed,
        step_length_k=state.alpha_star,
        func_kp1=state.phi_star,
        grad_kp1=state.grad_star,
        num_func_evals=state.num_func_evals,
        num_grad_evals=state.num_grad_evals,
        status=status,
    )


class BracketState(NamedTuple):
    is_terminated: BoolLike
    is_zoom: BoolLike
    is_star_found: BoolLike
    is_failed: BoolLike
    alpha_prev: FloatLike
    phi_prev: FloatLike
    grad_phi_prev: FloatLike
    grad_prev: JNPArray
    alpha_lo: FloatLike
    phi_lo: FloatLike
    grad_phi_lo: FloatLike
    grad_lo: JNPArray
    alpha_hi: FloatLike
    phi_hi: FloatLike
    grad_phi_hi: FloatLike
    alpha_star: FloatLike
    phi_star: FloatLike
    grad_star: JNPArray
    iteration: int
    num_func_evals: int
    num_grad_evals: int


class BracketResults(NamedTuple):
    is_star_found: BoolLike
    is_failed: BoolLike
    is_max_iters_fail: BoolLike
    num_iters: int
    alpha_lo: FloatLike
    phi_lo: FloatLike
    grad_phi_lo: FloatLike
    grad_lo: JNPArray
    alpha_hi: FloatLike
    phi_hi: FloatLike
    grad_phi_hi: FloatLike
    alpha_star: FloatLike
    phi_star: FloatLike
    grad_star: JNPArray
    num_func_evals: int
    num_grad_evals: int


def _bracket(
    line_func: LineFunc,
    wolfe_cond_one: WolfeCondOne,
    wolfe_cond_two: WolfeCondTwo,
    alpha_init: float,
    phi_0: FloatLike,
    grad_phi_0: FloatLike,
    grad_0: JNPArray,
    max_iters: int,
) -> BracketResults:
    state = BracketState(
        is_terminated=False,
        is_zoom=False,
        is_star_found=False,
        is_failed=False,
        alpha_prev=0.0,
        phi_prev=phi_0,
        grad_phi_prev=grad_phi_0,
        grad_prev=grad_0,
        alpha_lo=0.0,
        phi_lo=phi_0,
        grad_phi_lo=grad_phi_0,
        grad_lo=grad_0,
        alpha_hi=0.0,
        phi_hi=phi_0,
        grad_phi_hi=grad_phi_0,
        alpha_star=0.0,
        phi_star=phi_0,
        grad_star=grad_0,
        iteration=0,  # In contrast, Nocedal and Wright start with 1.
        num_func_evals=0,
        num_grad_evals=0,
    )

    def run_one_iteration(state: BracketState) -> BracketState:
        # In contrast to Nocedal and Wright:
        # - No alpha_max is used
        # - alpha is always set at the beginning of the loop, not at the end.
        # Alpha is just doubled in each iteration as in jax.scipy. The maximum alpha is thus indirectly given by the maximum number of iterations.

        alpha_i = jnp.where(state.iteration == 0, alpha_init, 2.0 * state.alpha_prev)

        phi_i, grad_phi_i, grad_i = line_func(alpha_i)
        state = state._replace(
            num_func_evals=state.num_func_evals + 1,
            num_grad_evals=state.num_grad_evals + 1,
        )

        # Meaning of the endings lo and hi:
        # - alpha_lo is among all step length generated so far and sattisfying the sufficient decrease condition the one resulting in the smallest function value.

        # Determine lo and hi for zooming:
        # - is_zoom_with_prev_as_lo: If the first Wolfe condition is not satified, then the previous step length (alpha_prev) is the step length with the smallest function value. This must be true because of the second codition. According to the second condition, alpha_i was never greater than alpha_prev.
        # - is_zoom_with_i_as_lo: Since the first Wolfe condition is satisfied and alpha_i is less than alpha_prev, alph_i must be the step length with the smallest function value.
        is_zoom_with_prev_as_lo = (~wolfe_cond_one(alpha_i, phi_i)) | (
            (phi_i >= state.phi_prev) & (state.iteration > 1)
        )
        is_star_found = wolfe_cond_two(grad_phi_i) & (~is_zoom_with_prev_as_lo)
        is_zoom_with_i_as_lo = (
            (grad_phi_i >= 0.0) & (~is_zoom_with_prev_as_lo) & (~is_star_found)
        )

        state = state._replace(
            is_zoom=is_zoom_with_prev_as_lo | is_zoom_with_i_as_lo,
            is_star_found=is_star_found,
        )

        state = state._replace(
            **_binary_replace(
                is_zoom_with_prev_as_lo,
                state._asdict(),
                dict(
                    alpha_lo=state.alpha_prev,
                    phi_lo=state.phi_prev,
                    grad_phi_lo=state.grad_phi_prev,
                    grad_lo=state.grad_prev,
                    alpha_hi=alpha_i,
                    phi_hi=phi_i,
                    grad_phi_hi=grad_phi_i,
                ),
            )
        )
        state = state._replace(
            **_binary_replace(
                is_zoom_with_i_as_lo,
                state._asdict(),
                dict(
                    alpha_lo=alpha_i,
                    phi_lo=phi_i,
                    grad_phi_lo=grad_phi_i,
                    grad_lo=grad_i,
                    alpha_hi=state.alpha_prev,
                    phi_hi=state.phi_prev,
                    grad_phi_hi=state.grad_phi_prev,
                ),
            )
        )
        state = state._replace(
            **_binary_replace(
                is_star_found,
                state._asdict(),
                dict(
                    alpha_star=alpha_i,
                    phi_star=phi_i,
                    grad_star=grad_i,
                ),
            )
        )
        state = state._replace(
            alpha_prev=alpha_i,
            phi_prev=phi_i,
            grad_phi_prev=grad_phi_i,
            grad_prev=grad_i,
            iteration=state.iteration + 1,
        )
        state = state._replace(
            is_terminated=state.is_zoom | state.is_star_found | state.is_failed,
        )
        return state

    cond_func = lambda state: (~state.is_terminated) & (state.iteration < max_iters)
    state = jax.lax.while_loop(cond_func, run_one_iteration, state)

    is_max_iters_fail = ~state.is_terminated

    return BracketResults(
        is_star_found=state.is_star_found,
        is_failed=state.is_failed | is_max_iters_fail,
        is_max_iters_fail=is_max_iters_fail,
        num_iters=state.iteration,
        alpha_lo=state.alpha_lo,
        phi_lo=state.phi_lo,
        grad_phi_lo=state.grad_phi_lo,
        grad_lo=state.grad_lo,
        alpha_hi=state.alpha_hi,
        phi_hi=state.phi_hi,
        grad_phi_hi=state.grad_phi_hi,
        alpha_star=state.alpha_star,
        phi_star=state.phi_star,
        grad_star=state.grad_star,
        num_func_evals=state.num_func_evals,
        num_grad_evals=state.num_grad_evals,
    )


class ZoomState(NamedTuple):
    is_terminated: BoolLike
    is_star_found: BoolLike
    is_failed: BoolLike
    alpha_lo: FloatLike
    phi_lo: FloatLike
    grad_phi_lo: FloatLike
    alpha_hi: FloatLike
    phi_hi: FloatLike
    grad_phi_hi: JNPFloat
    alpha_rec: FloatLike
    phi_rec: FloatLike
    alpha_star: FloatLike
    phi_star: FloatLike
    grad_star: JNPArray
    iteration: int
    num_func_evals: int
    num_grad_evals: int


class ZoomResults(NamedTuple):
    is_star_found: BoolLike
    is_failed: BoolLike
    is_max_iters_fail: BoolLike
    num_iters: int
    alpha_star: FloatLike
    phi_star: FloatLike
    grad_star: JNPArray
    num_func_evals: int
    num_grad_evals: int


def _zoom(
    line_func: LineFunc,
    wolfe_cond_one: WolfeCondOne,
    wolfe_cond_two: WolfeCondTwo,
    alpha_lo: FloatLike,
    phi_lo: FloatLike,
    grad_phi_lo: FloatLike,
    grad_lo: JNPArray,
    alpha_hi: FloatLike,
    phi_hi: FloatLike,
    grad_phi_hi: JNPFloat,
    max_iters: int,
    is_pass_through: BoolLike,
) -> ZoomResults:
    state = ZoomState(
        is_terminated=False,
        is_star_found=False,
        is_failed=False,
        alpha_lo=alpha_lo,
        phi_lo=phi_lo,
        grad_phi_lo=grad_phi_lo,
        alpha_hi=alpha_hi,
        phi_hi=phi_hi,
        grad_phi_hi=grad_phi_hi,
        alpha_rec=(alpha_lo + alpha_hi) / 2.0,
        phi_rec=(phi_lo + phi_hi) / 2.0,
        # Use other default values for alpha_star, phi_star and grad_star as in jax.scipy.
        alpha_star=alpha_lo,
        phi_star=phi_lo,
        grad_star=grad_lo,
        iteration=0,
        num_func_evals=0,
        num_grad_evals=0,
    )

    # As per jax.scipy.
    safeguard_factor_cubic = 0.2
    safeguard_factor_quad = 0.1

    def run_one_iteration(state: ZoomState) -> ZoomState:
        # In jax.scipy: d_alpha = state.alpha_hi - state.alpha_lo
        alpha_min = jnp.minimum(state.alpha_lo, state.alpha_hi)
        alpha_max = jnp.maximum(state.alpha_lo, state.alpha_hi)
        d_alpha = alpha_max - alpha_min
        safeguard_cubic = safeguard_factor_cubic * d_alpha
        safeguard_quad = safeguard_factor_quad * d_alpha

        # This check cause the line search to stop if two consecutive alphas are indistinguishable in finite-precision arithmetic. Since in case of a stop the Wolfe conditions are not satisfied, the optimization stops.
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
            (state.iteration > 0)  # Because initial values for rec are not exact.
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
        alpha_j_bisection = _bisection(alpha_lo=state.alpha_lo, alpha_hi=state.alpha_hi)
        use_bisection = (~use_cubic) & (~use_quad)

        alpha_j = jnp.where(use_cubic, alpha_j_cubic, state.alpha_rec)
        alpha_j = jnp.where(use_quad, alpha_j_quad, alpha_j)
        alpha_j = jnp.where(use_bisection, alpha_j_bisection, alpha_j)

        phi_j, grad_phi_j, grad_j = line_func(alpha_j)
        state._replace(
            num_func_evals=state.num_func_evals + 1,
            num_grad_evals=state.num_grad_evals + 1,
        )

        # First Wolfe condition is violated or phi_j>phi_lo. alpha_lo remains unchanged.
        hi_to_j = (~wolfe_cond_one(alpha_j, phi_j)) | (phi_j >= state.phi_lo)
        # In addition to the first (j_is_hi==False), the second Wolfe condition is also satidfied.
        star_to_j = (~hi_to_j) & wolfe_cond_two(grad_phi_j)
        # Only the first Wolfe condiion (suffiecient decrease condition) is satisfied.
        lo_to_j = (~hi_to_j) & (~star_to_j)
        # alpha_hi is chosen depending on grad_alpha_lo which is grad_alpha_j since j_is_lo==True.
        hi_to_old_lo = lo_to_j & (grad_phi_j * (state.alpha_hi - state.alpha_lo) >= 0.0)

        state = state._replace(is_star_found=star_to_j)

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
        state = state._replace(
            is_terminated=state.is_star_found | state.is_failed,
        )
        return state

    cond_func = (
        lambda state: (~is_pass_through)
        & (~state.is_terminated)
        & (state.iteration < max_iters)
    )
    state = jax.lax.while_loop(cond_func, run_one_iteration, state)

    is_max_iters_fail = (~state.is_terminated) & (~is_pass_through)

    return ZoomResults(
        is_star_found=state.is_star_found,
        is_failed=state.is_failed | is_max_iters_fail,
        is_max_iters_fail=is_max_iters_fail,
        num_iters=state.iteration,
        alpha_star=state.alpha_star,
        phi_star=state.phi_star,
        grad_star=state.grad_star,
        num_func_evals=state.num_func_evals,
        num_grad_evals=state.num_grad_evals,
    )


def _minimize_cubic(
    alpha_lo: FloatLike,
    phi_lo: FloatLike,
    grad_phi_lo: FloatLike,
    alpha_hi: FloatLike,
    phi_hi: FloatLike,
    alpha_rec: FloatLike,
    phi_rec: FloatLike,
) -> FloatLike:
    # Implementation from J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. New York: Springer, 2006.
    d_alpha_hi = alpha_hi - alpha_lo
    d_alpha_rec = alpha_rec - alpha_lo
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
    entry_a, entry_b = _dot(arr_alphas, arr_phis) / denominator
    radicand = entry_b**2 - 3.0 * entry_a * grad_phi_lo
    alpha_min = alpha_lo + (-entry_b + jnp.sqrt(radicand)) / (3.0 * entry_a)
    return alpha_min


def _minimize_quadratic(
    alpha_lo: FloatLike,
    phi_lo: FloatLike,
    grad_phi_lo: FloatLike,
    alpha_hi: FloatLike,
    phi_hi: FloatLike,
) -> FloatLike:
    d_alpha = alpha_lo - alpha_hi
    a = ((phi_lo - phi_hi - grad_phi_lo * d_alpha)) / (-(d_alpha**2))
    b = grad_phi_lo - 2 * a * alpha_lo
    x_min = -(b / 2 * a)
    return x_min


def _bisection(alpha_lo: FloatLike, alpha_hi: FloatLike) -> FloatLike:
    return (alpha_lo + alpha_hi) / 2.0


def _binary_replace(
    should_replace: bool,
    original_dict: dict[str, Any],
    new_dict: dict[str, Any],
    keys: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    keys = keys or tuple(new_dict.keys())
    output_dict = dict()
    for key in keys:
        output_dict[key] = jnp.where(should_replace, new_dict[key], original_dict[key])
    return output_dict
