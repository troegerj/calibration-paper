# Standard library imports
from typing import Union

# Third library imports
import pandas as pd
import numpy as np

# Local library imports
from calibrationpinn.typeAliases import (
    PDDataType,
    PDDataFrame,
    PDSeries,
    NPFloatTuple,
)

ValueType = Union[tuple[int], tuple[float], tuple[str, ...], NPFloatTuple]


class TrainingLoggerRecord:
    def __init__(self, column_names: tuple[str, ...], data_type: PDDataType) -> None:
        self._column_names = column_names
        self._data_type = data_type
        self._log_entries: list[PDDataFrame] = []

    def log(self, values: ValueType) -> None:
        log_entry = dict(zip(self._column_names, values))
        self._log_entries.append(log_entry)

    def reduce_to_minimum(self) -> PDDataFrame:
        record = self.as_data_frame()
        minimum_entries_as_series = record.min(axis=0, skipna=True, numeric_only=True)
        return self._convert_series_to_transposed_data_frame(minimum_entries_as_series)

    def reduce_to_last(self) -> PDDataFrame:
        record = self.as_data_frame()
        return record.tail(1)

    def as_data_frame(self) -> PDDataFrame:
        return pd.DataFrame(
            self._log_entries, columns=self._column_names, dtype=self._data_type
        )

    def _convert_series_to_transposed_data_frame(self, series: PDSeries) -> PDDataFrame:
        return pd.DataFrame(series).transpose()
