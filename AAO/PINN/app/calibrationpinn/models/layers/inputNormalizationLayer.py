# Standard library imports

# Third-party imports
import haiku as hk

# Local library imports
from calibrationpinn.typeAliases import JNPArray


class InputNormalizationLayer(hk.Module):
    def __init__(self, min_input: JNPArray, max_input: JNPArray, name: str) -> None:
        super().__init__(name=name)
        self._min_input = min_input
        self._input_range = max_input - min_input

    def __call__(self, inputs: JNPArray) -> JNPArray:
        return (((inputs - self._min_input) / self._input_range) * 2) - 1
