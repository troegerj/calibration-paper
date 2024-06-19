# Standard library imports

# Third-party imports
import haiku as hk
import jax.numpy as jnp

# Local library imports
from calibrationpinn.typeAliases import JNPArray


class FakeNetUx(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_ux")
        self._constant = 1 / 2
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_ux",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * jnp.square(input[0]) * input[1]


class FakeNetUy(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_uy")
        self._constant = 1
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_uy",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * input[0] * jnp.square(input[1])


class FakeNetUxPlaneStrain(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_ux")
        self._constant = 1 / 10
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_ux",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * jnp.square(input[0]) * input[1]


class FakeNetUyPlaneStrain(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_uy")
        self._constant = -2 / 5
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_uy",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * input[0] * jnp.square(input[1])


class FakeNetUxPlaneStress(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_ux")
        self._constant = 8 / 39
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_ux",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * jnp.square(input[0]) * input[1]


class FakeNetUyPlaneStress(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="fake_net_uy")
        self._constant = -44 / 39
        self._constant_init = hk.initializers.Constant(self._constant)

    def __call__(self, input: JNPArray) -> JNPArray:
        # The input must be a single input
        constant = hk.get_parameter(
            name="constant_net_uy",
            shape=(1,),
            dtype=input.dtype,
            init=self._constant_init,
        )
        return constant * input[0] * jnp.square(input[1])
