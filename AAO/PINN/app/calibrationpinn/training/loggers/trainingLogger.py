# Standard library imports
from typing import Optional

# Third-party imports
import pandas as pd
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import PandasDataWriter
from calibrationpinn.settings import Settings
from calibrationpinn.training.loggers.trainingLoggerRecord import TrainingLoggerRecord
from calibrationpinn.typeAliases import PDDataFrame, NPFloatTuple


class TrainingLogger:
    def __init__(
        self,
        loss_names: tuple[str, ...],
        error_metric_names: tuple[str, ...],
        parameter_names: tuple[str, ...],
        pandas_data_writer: PandasDataWriter,
        settings: Settings,
        additional_log_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        data_type = settings.DATA_TYPE
        self._data_writer = pandas_data_writer
        self.epochs_record = TrainingLoggerRecord(("epoch",), np.int64)
        self.losses_record = TrainingLoggerRecord(loss_names, data_type)
        self.error_metrics_record = TrainingLoggerRecord(error_metric_names, data_type)
        self.parameters_record = TrainingLoggerRecord(parameter_names, data_type)
        self.training_times_record = TrainingLoggerRecord(("time",), data_type)
        self.additional_logs_record = None
        if additional_log_names:
            self.additional_logs_record = TrainingLoggerRecord(
                additional_log_names, np.str_
            )

    def log(
        self,
        epoch: int,
        losses: NPFloatTuple,
        errors: NPFloatTuple,
        parameters: NPFloatTuple,
        training_time: float,
        additional_logs: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.epochs_record.log((epoch,))
        self.losses_record.log(losses)
        self.error_metrics_record.log(errors)
        self.parameters_record.log(parameters)
        self.training_times_record.log((training_time,))
        if self.additional_logs_record and additional_logs:
            self.additional_logs_record.log(additional_logs)

    def get_last_entries(self) -> PDDataFrame:
        last_entries = []
        for record in self._list_all_records():
            last_entries.append(record.reduce_to_last())
        return pd.concat(last_entries, axis=1)

    def get_best_entries(self) -> PDDataFrame:
        best_entries = []
        for record in self._list_loss_metric_and_time_records():
            best_entries.append(record.reduce_to_minimum())
        return pd.concat(best_entries, axis=1)

    def print(self) -> None:
        composed_records = self._compose_all_records()
        print(composed_records)

    def print_info(self) -> None:
        composed_records = self._compose_all_records()
        print(composed_records.info())

    def save_as_csv(self, output_file_name: str, output_subdirectory_name: str) -> None:
        composed_records = self._compose_all_records()
        self._data_writer.write(
            composed_records, output_file_name, output_subdirectory_name, header=True
        )

    def as_data_frame(self) -> PDDataFrame:
        return self._compose_all_records()

    def _list_all_records(self) -> list[PDDataFrame]:
        all_records = [
            self.epochs_record,
            self.losses_record,
            self.error_metrics_record,
            self.parameters_record,
            self.training_times_record,
        ]
        if self.additional_logs_record:
            all_records.append(self.additional_logs_record)
        return all_records

    def _list_loss_metric_and_time_records(self) -> list[PDDataFrame]:
        return [
            self.losses_record,
            self.error_metrics_record,
            self.training_times_record,
        ]

    def _compose_all_records(self) -> PDDataFrame:
        all_records = [
            self.epochs_record.as_data_frame(),
            self.losses_record.as_data_frame(),
            self.error_metrics_record.as_data_frame(),
            self.parameters_record.as_data_frame(),
            self.training_times_record.as_data_frame(),
        ]
        if self.additional_logs_record:
            all_records.append(self.additional_logs_record.as_data_frame())
        return pd.concat(all_records, axis=1)
