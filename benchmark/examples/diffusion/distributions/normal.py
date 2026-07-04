import keras.ops as ops
import keras.utils as utils
from math import pi
from bayesflow.types import Tensor

@utils.register_keras_serializable("bayesflow.utils")
def normal_lpdf(x: Tensor, mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    lpdf = -0.5 * ops.square(x-mu) / ops.square(sigma)

    lpdf -= 0.5 * ops.log(2 * pi)
    lpdf -= ops.log(sigma) # 0.5 log(sigma^2)
    return lpdf

@utils.register_keras_serializable("bayesflow.utils")
def normal_lcdf(x: Tensor, mu: float = 0.0, sigma: float = 1.0) -> Tensor:
    x = (x-mu)/sigma
    x = x/ops.sqrt(2)

    return ops.log1p(ops.erf(x)) - ops.log(2)
