# Standard library imports

# Third-party imports
import haiku as hk
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import HKInitializer, JNPArray


class ConstantFunction(hk.Module):
    def __init__(
        self, output_size: int, func_value_init: HKInitializer, name: str
    ) -> None:
        super().__init__(name=name)
        self._output_size = output_size
        self._func_value_init = func_value_init

    def __call__(self, input: JNPArray) -> JNPArray:
        func_value = hk.get_parameter(
            name="func_value",
            shape=(self._output_size,),
            dtype=input.dtype,
            init=self._func_value_init,
        )
        input_size = input.shape[-1]
        output_size = self._output_size
        zero_weights = jnp.zeros((input_size, output_size))
        return jnp.dot(input, zero_weights) + func_value
