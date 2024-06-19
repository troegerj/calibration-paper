# Standard library imports
from typing import Callable

# Third-party imports
import haiku as hk

# Local library imports
from calibrationpinn.typeAliases import HKInitializer, JNPArray


class ActivatedLayer(hk.Module):
    def __init__(
        self,
        output_size: int,
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
        name: str,
    ) -> None:
        super().__init__(name=name)
        self._linear_transformation = hk.Linear(
            output_size=output_size,
            with_bias=True,
            w_init=w_init,
            b_init=b_init,
            name="linear_transformation",
        )
        self._activation = activation

    def __call__(self, input: JNPArray) -> JNPArray:
        return self._activation(self._linear_transformation(input))
