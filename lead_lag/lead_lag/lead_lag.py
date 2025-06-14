from collections import deque
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd

from .lead_lag_impl import CrossCorrelationHY


# noinspection PyBroadException
def prune_to_specific_precision(d: pd.Series, target_precision_ms) -> pd.Series:
    try:
        return d[0:-1][
            np.array(np.diff(d.index) / 1e6, dtype=int) > target_precision_ms * 1e3
        ]
    except Exception:
        return d[0:-1][
            np.array(
                [
                    a.total_seconds() * 1e3 > target_precision_ms
                    for a in np.diff(d.index)
                ]
            )
        ]


class LeadLag:

    def __init__(
        self,
        ts1: pd.Series,
        ts2: pd.Series,
        max_lag: Union[
            float, int
        ],  # in seconds. Interval of research: [-max_lag, +max_lag].
        verbose: bool = False,
        min_precision: Optional[float] = None,
        specific_lags: Optional[List[float]] = None,
    ):
        self.verbose = verbose
        ts1.dropna(inplace=True)
        ts2.dropna(inplace=True)
        ts1.sort_index(inplace=True)
        ts2.sort_index(inplace=True)
        ts1 = ts1[~ts1.index.duplicated(keep="first")]
        ts2 = ts2[~ts2.index.duplicated(keep="first")]
        data_precision = min(
            min(
                [
                    (ts1.index[i] - ts1.index[i - 1]).total_seconds()
                    for i in range(1, len(ts1))
                ]
            ),
            min(
                [
                    (ts2.index[i] - ts2.index[i - 1]).total_seconds()
                    for i in range(1, len(ts2))
                ]
            ),
        )
        if min_precision is None:
            min_precision = data_precision
        if data_precision < min_precision:
            # nano -> ms (/1e6). precision*1e3 -> ms.
            dp1 = len(ts1) + len(ts2)
            print(
                f"WARNING: The data provided has a precision of {data_precision * 1e3} ms. "
                f"The minimum precision specified is {min_precision * 1e3} ms. "
                f"To reach this precision, some data point are to be discarded."
            )
            ts1 = prune_to_specific_precision(ts1, min_precision)
            ts2 = prune_to_specific_precision(ts2, min_precision)
            dp2 = len(ts1) + len(ts2)
            print(
                f"WARNING: {dp1 - dp2} data points were discarded. "
                f"This represents {100 * (dp1 - dp2) / dp1:.5f}% of the data."
            )
        if min_precision not in {1, 0.1, 0.01, 0.001, 0.0001}:
            if 0.1 < min_precision < 1:
                min_precision = 0.1
            elif 0.01 < min_precision < 0.1:
                min_precision = 0.01
            elif 0.001 < min_precision < 0.01:
                min_precision = 0.001
            elif 0.0001 < min_precision < 0.001:
                min_precision = 0.0001
            else:
                raise Exception(
                    "Valid values for precision are 1, 0.1, 0.01, 0.001 and 0.0001. Minimum is 100us."
                )
        self.contrasts = None
        self.precision = min_precision
        exponents = dict({1: 9, 0.1: 8, 0.01: 7, 0.001: 6, 0.0001: 5})
        t1 = ts1.index.values.astype(np.int64) // 10 ** exponents[self.precision]
        t2 = ts2.index.values.astype(np.int64) // 10 ** exponents[self.precision]
        arr_1_with_ts = np.stack([t1, ts1.values], axis=1)
        arr_2_with_ts = np.stack([t2, ts2.values], axis=1)
        (
            self.x,
            self.y,
            self.t_x,
            self.t_y,
        ) = convert_to_lead_lag_format(arr_1_with_ts, arr_2_with_ts)
        assert len(self.x) == len(self.y)
        max_lag = int(max_lag / self.precision)
        if max_lag <= 0:
            raise Exception(
                "Max lag is too low. Increase it. Or increase the precision."
            )
        if specific_lags is None:
            self.lag_range = np.arange(-max_lag, max_lag + 1, 1)
        else:
            self.lag_range = np.array(sorted(specific_lags))
        self.inference_time = None
        self.cc = CrossCorrelationHY(
            self.x,
            self.y,
            self.t_x,
            self.t_y,
            self.lag_range,
            normalize=True,
            verbose=self.verbose,
        )

    def run_inference(self, num_threads: int = 5):
        start_time = time()
        self.contrasts = self.cc.fast_inference(num_threads)
        self.inference_time = time() - start_time

    @property
    def lead_lag(self) -> Optional[float]:
        if self.contrasts is None:
            return None
        if np.std(self.contrasts) == 0.0:
            return None
        return self.lag_range[np.argmax(self.contrasts)] * self.precision

    @property
    def llr(self) -> Optional[float]:
        llr = np.nan
        if self.contrasts is None:
            return None
        positive_range_indexes = self.lag_range > 0
        negative_range_indexes = self.lag_range < 0
        positive_contrasts = np.sum(self.contrasts[positive_range_indexes])
        negative_contrasts = np.sum(self.contrasts[negative_range_indexes]) + 0.1**10

        if negative_contrasts != 0.0:
            llr = positive_contrasts / negative_contrasts
        return llr

    @property
    def llr2(self) -> Optional[float]:
        llr2 = np.nan
        if self.contrasts is None:
            return None
        positive_range_indexes = self.lag_range > 0
        negative_range_indexes = self.lag_range < 0
        positive_contrasts2 = np.sum(self.contrasts[positive_range_indexes] ** 2)
        negative_contrasts2 = (
            np.sum(self.contrasts[negative_range_indexes] ** 2) + 0.1**10
        )
        if negative_contrasts2 != 0.0:
            llr2 = positive_contrasts2 / negative_contrasts2
        return llr2


def convert_to_lead_lag_format(
    arr1: np.array, arr2: np.array
) -> Tuple[np.array, np.array, np.array, np.array]:
    assert len(arr1.shape) == 2  # (x, t_x)
    assert len(arr2.shape) == 2  # (y, t_y)
    time_origin = min(arr2[0, 0], arr1[0, 0])
    arr1[:, 0] -= time_origin
    arr2[:, 0] -= time_origin
    time_end = int(max(arr2[-1, 0], arr1[-1, 0]))
    x = np.zeros(shape=time_end + 1) * np.nan
    t_x = []
    for element_slice in arr1:
        x[int(element_slice[0])] = element_slice[1]
        t_x.append(int(element_slice[0]))
    y = np.zeros(shape=time_end + 1) * np.nan
    t_y = []
    for element_slice in arr2:
        y[int(element_slice[0])] = element_slice[1]
        t_y.append(int(element_slice[0]))
    return x, y, t_x, t_y
