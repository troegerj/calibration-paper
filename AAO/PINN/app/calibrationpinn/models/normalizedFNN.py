# Standard library imports
from typing import Callable

# Third-party imports
import haiku as hk
import jax.numpy as jnp

# Local library imports
from calibrationpinn.models.layers import (
    ActivatedLayer,
    InputNormalizationLayer,
    OutputRenormalizationLayer,
)
from calibrationpinn.typeAliases import HKInitializer, HKModule, JNPArray


class NormalizedFNN(hk.Module):
    def __init__(
        self,
        output_sizes: list[int],
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
        reference_inputs: JNPArray,
        reference_outputs: JNPArray,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self._layers = self._initialize_layers(
            output_sizes,
            w_init,
            b_init,
            activation,
            reference_inputs,
            reference_outputs,
        )

    def _initialize_layers(
        self,
        output_sizes: list[int],
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
        reference_inputs: JNPArray,
        reference_outputs: JNPArray,
    ) -> tuple[HKModule, ...]:
        layers = []
        layers.append(self._init_input_normalization_layer(reference_inputs))
        layers.extend(
            self._init_hidden_layers(output_sizes[:-1], w_init, b_init, activation)
        )
        layers.append(self._init_output_layer(output_sizes[-1], w_init, b_init))
        layers.append(self._init_output_renormalization_layer(reference_outputs))
        return tuple(layers)

    def _init_input_normalization_layer(self, reference_inputs: JNPArray) -> HKModule:
        min_input = jnp.amin(reference_inputs, axis=0)
        max_input = jnp.amax(reference_inputs, axis=0)
        return InputNormalizationLayer(
            min_input, max_input, name="input_normalization_layer"
        )

    def _init_hidden_layers(
        self,
        output_sizes: list[int],
        w_init: HKInitializer,
        b_init: HKInitializer,
        activation: Callable[[JNPArray], JNPArray],
    ) -> list[HKModule]:
        layers = []
        for index, output_size in enumerate(output_sizes):
            layers.append(
                ActivatedLayer(
                    output_size=output_size,
                    w_init=w_init,
                    b_init=b_init,
                    activation=activation,
                    name="hidden_layer_{number}".format(number=index),
                )
            )
        return layers

    def _init_output_layer(
        self, output_size: int, w_init: HKInitializer, b_init: HKInitializer
    ) -> HKModule:
        return hk.Linear(
            output_size=output_size,
            with_bias=True,
            w_init=w_init,
            b_init=b_init,
            name="output_layer",
        )

    def _init_output_renormalization_layer(
        self, reference_outputs: JNPArray
    ) -> HKModule:
        min_output = jnp.amin(reference_outputs, axis=0)
        max_output = jnp.amax(reference_outputs, axis=0)
        return OutputRenormalizationLayer(
            min_output, max_output, name="output_renormalization_layer"
        )

    def __call__(self, input: JNPArray) -> JNPArray:
        output = input
        for layer in self._layers:
            output = layer(output)
        return output
