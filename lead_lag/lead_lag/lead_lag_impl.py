import numpy as np
from bisect import bisect_left
import math

"""
Used for Python. This file will not be converted to C code.
"""

import os
from time import time


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool

    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned


def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2)) > 0


def l2_norm_of_arr_diff(x):
    return np.linalg.norm(np.diff(x[~np.isnan(x)])) + 1e-8  # for numerical stability.


class CrossCorrelationHY:

    def __init__(self, x, y, t_x, t_y, lag_range, normalize=True, verbose=True):
        self.x = np.array(x)
        self.y = np.array(y)
        self.t_x = np.array(t_x)
        self.t_y = np.array(t_y)
        self.lag_range = lag_range
        self.normalize = normalize
        self.verbose = verbose

    def fast_inference(self, num_threads=int(os.cpu_count())):
        if self.verbose:
            print(
                f"Running fast_inference() on {len(self.lag_range)} lags with {num_threads} threads."
            )
        contrast = parallel_function(self.call, self.lag_range, num_threads=num_threads)
        return np.array(contrast)

    def call(self, k):
        start_time = time()
        value = shifted_modified_hy_estimator(
            self.x, self.y, self.t_x, self.t_y, k, self.normalize
        )
        end_time = time()
        if self.verbose:
            print(
                f"Lag={k}, contrast={value:.5f}, elapsed={(end_time - start_time) * 1e3:.2f}ms."
            )
        return value


def shifted_modified_hy_estimator(
    x, y, t_x, t_y, k, normalize=False
):  # contrast function
    hy_cov = 0.0
    norm_x = 1.0
    norm_y = 1.0

    if normalize:
        norm_x = l2_norm_of_arr_diff(x)
        norm_y = l2_norm_of_arr_diff(y)

    clipped_t_y_minus_k = np.clip(np.array(t_y) - k, int(np.min(t_y)), int(np.max(t_y)))
    # Complexity: O(n log n)
    for ii0, ii1 in zip(t_x, t_x[1:]):  # O(n)
        x_inc = x[ii1] - x[ii0]
        mid_point_origin = bisect_left(clipped_t_y_minus_k, ii0)  # O(log n)
        if mid_point_origin is None:
            raise Exception(" mid_point_origin is None")
        # go left
        mid_point = mid_point_origin
        while True:
            if mid_point + 1 >= len(t_y) or mid_point < 0:
                break
            jj0, jj1 = (t_y[mid_point], t_y[mid_point + 1])
            if overlap(ii0, ii1, jj0 - k, jj1 - k):
                hy_cov += (y[jj1] - y[jj0]) * x_inc
                mid_point += 1
            else:
                break
        # go right
        mid_point = mid_point_origin - 1
        while True:
            if mid_point + 1 >= len(t_y) or mid_point < 0:
                break
            jj0, jj1 = (t_y[mid_point], t_y[mid_point + 1])
            if overlap(ii0, ii1, jj0 - k, jj1 - k):
                hy_cov += (y[jj1] - y[jj0]) * x_inc
                mid_point -= 1
            else:
                break

    return np.abs(hy_cov) / (norm_x * norm_y)  # product of norm is positive.
