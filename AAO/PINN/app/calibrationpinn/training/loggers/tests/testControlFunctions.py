# Standard library imports
import unittest

# Third-party imports


# Local library imports
from calibrationpinn.assertions import assert_true, assert_false
from calibrationpinn.training.loggers import should_model_be_validated


class TestShouldModelBeValidatedFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._number_epochs = 11
        self._validation_interval = 10
        self._sut = should_model_be_validated

    def test_func_for_epoch_1_number_of_epochs_11_validation_interval_10(self) -> None:
        """
        Test that the function correctly determines whether the model should be validated or not.
        """
        epoch = 1
        should_model_be_validated = self._sut(
            epoch=epoch,
            num_epochs=self._number_epochs,
            valid_interval=self._validation_interval,
        )

        assert_true(self=self, expression=should_model_be_validated)

    def test_func_for_epoch_2_number_of_epochs_11_validation_interval_10(self) -> None:
        """
        Test that the function correctly determines whether the model should be validated or not.
        """
        epoch = 2
        should_model_be_validated = self._sut(
            epoch=epoch,
            num_epochs=self._number_epochs,
            valid_interval=self._validation_interval,
        )

        assert_false(self=self, expression=should_model_be_validated)

    def test_func_for_epoch_10_number_of_epochs_11_validation_interval_10(self) -> None:
        """
        Test that the function correctly determines whether the model should be validated or not.
        """
        epoch = 10
        should_model_be_validated = self._sut(
            epoch=epoch,
            num_epochs=self._number_epochs,
            valid_interval=self._validation_interval,
        )

        assert_true(self=self, expression=should_model_be_validated)

    def test_func_for_epoch_11_number_of_epochs_11_validation_interval_10(self) -> None:
        """
        Test that the function correctly determines whether the model should be validated or not.
        """
        epoch = 11
        should_model_be_validated = self._sut(
            epoch=epoch,
            num_epochs=self._number_epochs,
            valid_interval=self._validation_interval,
        )

        assert_true(self=self, expression=should_model_be_validated)
