# Standard library imports

# Third-party imports
import haiku as hk
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray


class FakeNetU(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_u")
        self._constant = 2.0
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_u",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * jnp.square(input[0])
