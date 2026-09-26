import keras.ops as ops
import keras.utils as utils
from math import pi
from bayesflow.types import Tensor
from .normal import normal_lcdf
from .utils import log1m_exp


@utils.register_keras_serializable("bayesflow.utils")
def wald_lpdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    lpdf = (
        ops.log(alpha)
        - 0.5 * ops.log(2 * pi)
        - 1.5 * ops.log(x)
        - ops.square(alpha - nu * x) / (2 * x)
    )

    return lpdf


@utils.register_keras_serializable("bayesflow.utils")
def wald_lcdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    ax = ops.divide(alpha, ops.sqrt(x))
    xva = ops.divide(ops.multiply(x, nu), alpha)
    av = 2.0 * ops.multiply(alpha, nu)

    term0 = normal_lcdf(ax * (xva - 1))
    term1 = av + normal_lcdf(-ax * (xva + 1))

    lcdf = ops.logaddexp(term0, term1)

    return lcdf


@utils.register_keras_serializable("bayesflow.utils")
def wald_lccdf(x: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
    lcdf = wald_lcdf(x=x, alpha=alpha, nu=nu)
    return log1m_exp(lcdf)


@utils.register_keras_serializable("bayesflow.utils")
def rdm_lpdf(rt: Tensor, alpha: Tensor, nu: Tensor, tau: Tensor) -> Tensor:

    # rt.shape = (batch_size, num_trials)
    # alpha.shape = (batch_size, num_trials, 2)
    # nu.shape = (batch_size, 1, 2)
    # tau.shape = (batch_size, 1)

    # non-decision time
    t0 = tau * ops.min(ops.abs(rt), axis=-1, keepdims=True)
    # decision time
    t = ops.abs(rt) - t0

    lpdf = ops.where(
        rt > 0,
        wald_lpdf(x=t, alpha=alpha[..., 0], nu=nu[..., 0]),
        wald_lpdf(x=t, alpha=alpha[..., 1], nu=nu[..., 1]),
    )

    lccdf = ops.where(
        rt < 0,
        wald_lccdf(x=t, alpha=alpha[..., 0], nu=nu[..., 0]),
        wald_lccdf(x=t, alpha=alpha[..., 1], nu=nu[..., 1]),
    )

    result = lpdf + lccdf

    # remove underflow (too low densities)
    result = ops.nan_to_num(result, nan=-100, posinf=0, neginf=-100)

    return result
