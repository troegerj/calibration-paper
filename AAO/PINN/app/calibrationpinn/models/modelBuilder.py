# Standard library imports
from typing import Callable

# Third-party imports
import haiku as hk

# Local library imports
from calibrationpinn.models.constantFunction import ConstantFunction
from calibrationpinn.models.normalizedFNN import NormalizedFNN
from calibrationpinn.typeAliases import HKInitializer, HKTransformed, JNPArray


class ModelBuilder:
    def build_feedforward_neural_network(
        self,
        output_sizes: list[int],
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
        name: str,
    ) -> HKTransformed:
        def model_func(input: JNPArray) -> JNPArray:
            model = hk.nets.MLP(
                output_sizes=output_sizes,
                w_init=w_init,
                b_init=b_init,
                with_bias=True,
                activation=activation,
                activate_final=False,
                name=name,
            )
            return model(input)

        return hk.without_apply_rng(hk.transform(model_func))

    def build_normalized_feedforward_neural_network(
        self,
        output_sizes: list[int],
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
        reference_inputs: JNPArray,
        reference_outputs: JNPArray,
        name: str,
    ) -> HKTransformed:
        def model_func(input: JNPArray) -> JNPArray:
            model = NormalizedFNN(
                output_sizes=output_sizes,
                w_init=w_init,
                b_init=b_init,
                activation=activation,
                reference_inputs=reference_inputs,
                reference_outputs=reference_outputs,
                name=name,
            )
            return model(input)

        return hk.without_apply_rng(hk.transform(model_func))

    def build_constant_function(
        self, output_size: int, func_value_init: HKInitializer, name: str
    ) -> HKTransformed:
        def model_func(input: JNPArray) -> JNPArray:
            model = ConstantFunction(
                output_size=output_size, func_value_init=func_value_init, name=name
            )
            return model(input)

        return hk.without_apply_rng(hk.transform(model_func))
