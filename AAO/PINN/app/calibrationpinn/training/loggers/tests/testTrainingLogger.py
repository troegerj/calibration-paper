# Standard library imports
import unittest

# Third-party imports
import pandas as pd
import numpy as np


# Local library imports
from calibrationpinn.assertions import (
    assert_equal_data_frames,
    assert_equal_data_frames_without_index,
)
from calibrationpinn.inputoutput import PathAdministrator, PandasDataWriter
from calibrationpinn.settings import Settings
from calibrationpinn.training.loggers import TrainingLogger


class TestTrainingLoggerWithoutAdditionalLogs(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        settings = Settings()
        path_admin = PathAdministrator(settings)
        pandas_data_writer = PandasDataWriter(path_admin)
        self._epoch_name = ("epoch",)
        self._loss_names = ("loss_1", "loss_2")
        self._error_metric_names = ("error_metric_1", "error_metric_2")
        self._parameter_names = ("parameter_1", "parameter_2")
        self._training_time_name = ("time",)
        self._sut = TrainingLogger(
            loss_names=self._loss_names,
            error_metric_names=self._error_metric_names,
            parameter_names=self._parameter_names,
            pandas_data_writer=pandas_data_writer,
            settings=settings,
        )
        self._data_type = settings.DATA_TYPE
        self._epoch_update_1 = 1
        self._losses_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._error_metrics_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._parameters_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._training_time_update_1 = 1.0
        self._epoch_update_2 = 2
        self._losses_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._error_metrics_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._parameters_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._training_time_update_2 = 2.0
        self._sut.log(
            epoch=self._epoch_update_1,
            losses=self._losses_update_1,
            errors=self._error_metrics_update_1,
            parameters=self._parameters_update_1,
            training_time=self._training_time_update_1,
        )
        self._sut.log(
            epoch=self._epoch_update_2,
            losses=self._losses_update_2,
            errors=self._error_metrics_update_2,
            parameters=self._parameters_update_2,
            training_time=self._training_time_update_2,
        )

    def test_log(self) -> None:
        """
        Test that the results are logged correctly and that the values returned by the training logger are equal to the results which are logged before.
        """
        actual = self._sut.as_data_frame()

        epoch = pd.DataFrame(
            np.array([self._epoch_update_1, self._epoch_update_2], dtype=np.int64),
            columns=self._epoch_name,
        )
        losses = pd.DataFrame(
            np.array(
                [self._losses_update_1, self._losses_update_2], dtype=self._data_type
            ),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array(
                [self._error_metrics_update_1, self._error_metrics_update_2],
                dtype=self._data_type,
            ),
            columns=self._error_metric_names,
        )
        parameters = pd.DataFrame(
            np.array(
                [self._parameters_update_1, self._parameters_update_2],
                dtype=self._data_type,
            ),
            columns=self._parameter_names,
        )
        training_time = pd.DataFrame(
            np.array(
                [self._training_time_update_1, self._training_time_update_2],
                dtype=self._data_type,
            ),
            columns=self._training_time_name,
        )
        expected = pd.concat([epoch, losses, errors, parameters, training_time], axis=1)
        assert_equal_data_frames(self=self, expected=expected, actual=actual)

    def test_get_last_entries(self) -> None:
        """
        Test that the last entries logged are returned correctly.
        """
        actual = self._sut.get_last_entries()

        epoch = pd.DataFrame(
            np.array([self._epoch_update_2], dtype=np.int64), columns=self._epoch_name
        )
        losses = pd.DataFrame(
            np.array([self._losses_update_2], dtype=self._data_type),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array([self._error_metrics_update_2], dtype=self._data_type),
            columns=self._error_metric_names,
        )
        parameters = pd.DataFrame(
            np.array([self._parameters_update_2], dtype=self._data_type),
            columns=self._parameter_names,
        )
        training_time = pd.DataFrame(
            np.array([self._training_time_update_2], dtype=self._data_type),
            columns=self._training_time_name,
        )
        expected = pd.concat([epoch, losses, errors, parameters, training_time], axis=1)
        assert_equal_data_frames_without_index(
            self=self, expected=expected, actual=actual
        )

    def test_get_best_entries(self) -> None:
        """
        Test that the best entries logged are returned correctly.
        """
        actual = self._sut.get_best_entries()

        losses = pd.DataFrame(
            np.array([self._losses_update_1], dtype=self._data_type),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array([self._error_metrics_update_1], dtype=self._data_type),
            columns=self._error_metric_names,
        )
        training_time = pd.DataFrame(
            np.array([self._training_time_update_1], dtype=self._data_type),
            columns=self._training_time_name,
        )
        expected = pd.concat([losses, errors, training_time], axis=1)
        assert_equal_data_frames_without_index(
            self=self, expected=expected, actual=actual
        )


class TestTrainingLoggerWithAdditionalLogs(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        settings = Settings()
        path_admin = PathAdministrator(settings)
        pandas_data_writer = PandasDataWriter(path_admin)
        self._epoch_name = ("epoch",)
        self._loss_names = ("loss_1", "loss_2")
        self._error_metric_names = ("error_metric_1", "error_metric_2")
        self._parameter_names = ("parameter_1", "parameter_2")
        self._training_time_name = ("time",)
        self._additional_log_names = ("additional_log_1", "additional_log_2")
        self._sut = TrainingLogger(
            loss_names=self._loss_names,
            error_metric_names=self._error_metric_names,
            parameter_names=self._parameter_names,
            pandas_data_writer=pandas_data_writer,
            settings=settings,
            additional_log_names=self._additional_log_names,
        )
        self._data_type = settings.DATA_TYPE
        self._epoch_update_1 = 1
        self._losses_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._error_metrics_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._parameters_update_1 = (self._data_type(1.0), self._data_type(1.0))
        self._training_time_update_1 = 1.0
        self._additional_logs_update_1 = (str(1.0), str(1.0))
        self._epoch_update_2 = 2
        self._losses_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._error_metrics_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._parameters_update_2 = (self._data_type(2.0), self._data_type(2.0))
        self._training_time_update_2 = 2.0
        self._additional_logs_update_2 = (str(2.0), str(2.0))
        self._sut.log(
            epoch=self._epoch_update_1,
            losses=self._losses_update_1,
            errors=self._error_metrics_update_1,
            parameters=self._parameters_update_1,
            training_time=self._training_time_update_1,
            additional_logs=self._additional_logs_update_1,
        )
        self._sut.log(
            epoch=self._epoch_update_2,
            losses=self._losses_update_2,
            errors=self._error_metrics_update_2,
            parameters=self._parameters_update_2,
            training_time=self._training_time_update_2,
            additional_logs=self._additional_logs_update_2,
        )

    def test_log(self) -> None:
        """
        Test that the results are logged correctly and that the values returned by the training logger are equal to the results which are logged before.
        """
        actual = self._sut.as_data_frame()

        epoch = pd.DataFrame(
            np.array([self._epoch_update_1, self._epoch_update_2], dtype=np.int64),
            columns=self._epoch_name,
        )
        losses = pd.DataFrame(
            np.array(
                [self._losses_update_1, self._losses_update_2], dtype=self._data_type
            ),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array(
                [self._error_metrics_update_1, self._error_metrics_update_2],
                dtype=self._data_type,
            ),
            columns=self._error_metric_names,
        )
        parameters = pd.DataFrame(
            np.array(
                [self._parameters_update_1, self._parameters_update_2],
                dtype=self._data_type,
            ),
            columns=self._parameter_names,
        )
        training_time = pd.DataFrame(
            np.array(
                [self._training_time_update_1, self._training_time_update_2],
                dtype=self._data_type,
            ),
            columns=self._training_time_name,
        )
        additional_logs = pd.DataFrame(
            np.array(
                [self._additional_logs_update_1, self._additional_logs_update_2],
                dtype=np.str_,
            ),
            columns=self._additional_log_names,
        )
        expected = pd.concat(
            [epoch, losses, errors, parameters, training_time, additional_logs],
            axis=1,
        )

        assert_equal_data_frames(self=self, expected=expected, actual=actual)

    def test_get_last_entries(self) -> None:
        """
        Test that the last entries logged are returned correctly.
        """
        actual = self._sut.get_last_entries()

        epoch = pd.DataFrame(
            np.array([self._epoch_update_2], dtype=np.int64), columns=self._epoch_name
        )
        losses = pd.DataFrame(
            np.array([self._losses_update_2], dtype=self._data_type),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array([self._error_metrics_update_2], dtype=self._data_type),
            columns=self._error_metric_names,
        )
        parameters = pd.DataFrame(
            np.array([self._parameters_update_2], dtype=self._data_type),
            columns=self._parameter_names,
        )
        training_time = pd.DataFrame(
            np.array([self._training_time_update_2], dtype=self._data_type),
            columns=self._training_time_name,
        )
        additional_logs = pd.DataFrame(
            np.array([self._additional_logs_update_2], dtype=np.str_),
            columns=self._additional_log_names,
        )
        expected = pd.concat(
            [epoch, losses, errors, parameters, training_time, additional_logs],
            axis=1,
        )
        assert_equal_data_frames_without_index(
            self=self, expected=expected, actual=actual
        )

    def test_get_best_entries(self) -> None:
        """
        Test that the best entries logged are returned correctly.
        """
        actual = self._sut.get_best_entries()

        losses = pd.DataFrame(
            np.array([self._losses_update_1], dtype=self._data_type),
            columns=self._loss_names,
        )
        errors = pd.DataFrame(
            np.array([self._error_metrics_update_1], dtype=self._data_type),
            columns=self._error_metric_names,
        )
        training_time = pd.DataFrame(
            np.array([self._training_time_update_1], dtype=self._data_type),
            columns=self._training_time_name,
        )
        expected = pd.concat([losses, errors, training_time], axis=1)
        assert_equal_data_frames_without_index(
            self=self, expected=expected, actual=actual
        )
