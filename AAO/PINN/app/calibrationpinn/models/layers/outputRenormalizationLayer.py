# Standard library imports

# Third-party imports
import haiku as hk

# Local library imports
from calibrationpinn.typeAliases import JNPArray


class OutputRenormalizationLayer(hk.Module):
    def __init__(self, min_output: JNPArray, max_output: JNPArray, name: str) -> None:
        super().__init__(name=name)
        self._min_output = min_output
        self._output_range = max_output - min_output

    def __call__(self, outputs: JNPArray) -> JNPArray:
        return (((outputs + 1) / 2) * self._output_range) + self._min_output
