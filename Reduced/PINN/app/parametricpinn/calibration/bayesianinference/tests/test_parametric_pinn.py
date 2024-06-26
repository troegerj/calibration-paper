import math

import torch

from parametricpinn.ansatz.base import (
    AnsatzStrategy,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.calibration.base import PreprocessedCalibrationData
from parametricpinn.calibration.bayesianinference.parametric_pinn import PPINNLikelihood
from parametricpinn.network import FFNN
from parametricpinn.types import Tensor

device = torch.device("cpu")


class FakeAnsatzStrategy(AnsatzStrategy):
    def __call__(self, x: Tensor, network: Networks) -> Tensor:
        return torch.zeros((1,))


class FakeAnsatz_SingleDimension(StandardAnsatz):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__(network, ansatz_strategy)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=1, keepdim=True)


class FakeAnsatz_MultipleDimension(StandardAnsatz):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__(network, ansatz_strategy)

    def forward(self, x: Tensor) -> Tensor:
        return torch.concat(
            (torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True)),
            dim=1,
        )


def _create_fake_ansatz_single_dimension() -> StandardAnsatz:
    fake_network = FFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeAnsatz_SingleDimension(fake_network, fake_ansatz_strategy)


def _create_fake_ansatz_multiple_dimension() -> StandardAnsatz:
    fake_network = FFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeAnsatz_MultipleDimension(fake_network, fake_ansatz_strategy)


def test_calibration_likelihood_single_data_single_dimension():
    ansatz = _create_fake_ansatz_single_dimension()
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = std_noise**2
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    sut = PPINNLikelihood(
        ansatz=ansatz,
        data=data,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1 / torch.sqrt(2 * torch.tensor(math.pi) * covariance_error)
    ) * torch.pow(torch.tensor(math.e), -1).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_calibration_likelihood_multiple_data_single_dimension():
    ansatz = _create_fake_ansatz_single_dimension()
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    sut = PPINNLikelihood(
        ansatz=ansatz,
        data=data,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_calibration_likelihood_single_data_multiple_dimension():
    model = _create_fake_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    sut = PPINNLikelihood(
        ansatz=model,
        data=data,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_calibration_likelihood_multiple_data_multiple_dimension():
    model = _create_fake_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((4,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    sut = PPINNLikelihood(
        ansatz=model,
        data=data,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (
            torch.pow((2 * torch.tensor(math.pi)), 2)
            * torch.sqrt(torch.det(covariance_error))
        )
        * torch.pow(torch.tensor(math.e), -4)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)
