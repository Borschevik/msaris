"""
    Various tools for spectre preprocessing
"""
from bisect import (
    bisect_left,
    bisect_right,
)
from typing import (
    Any,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def z_score(x: pd.Series, window: int) -> pd.Series:
    """
    performing z-score calculations with defined moving window

    :param x: pandas series with
    :param window: moving window for selected serie

    :return: value of z score for defined window
    """
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    return z


def z_score_filtering(
    mz: list, it: list, quantile: float = 0.99, window: Optional[int] = None
) -> np.array:
    """
    Perform simple z_score filtering for selected mass spectrum
    :param mz: m/z of
    :param it: list experiment intensity
    :param quantile: float quantile which defines z-score above which peaks would be retained
    :param window: int for z-score filter

    :returns: list of filtered intensities
    """

    if window is None:
        min_mz, max_mz = int(min(mz) + 0.5), int(max(mz) + 0.5)
        resolution = []
        for i in range(min_mz, max_mz, 10):
            resolution.append(bisect_right(mz, i + 10) - bisect_left(mz, i))
        window = int(np.percentile(resolution, 20) + 0.5)

    it_copy = it.copy()
    it_transformed: pd.Series = pd.Series(it_copy)
    zscores = z_score(it_transformed, window)
    threshold = zscores.quantile(quantile)
    it_transformed[zscores < threshold] = 0
    return it_transformed.to_numpy()


def reduce_data_resolution(
    mz_r: np.ndarray, it_r: np.ndarray, out_x: int, out_y: int
) -> Tuple[Any, Any]:
    """
    Reduction of the resolution by using interpolation can be used for high-definition data
    Using regular regrid to reduce resolution of spectra
    :param mz_r: list of m/z values
    :param it_r: list of intensities
    :param out_x: resulted number of values for m/z
    :param out_y: resulted number of falues for intensities

    :return: interpolated data with reduced resolution
    """
    m = mz_r.shape[0]
    y = np.linspace(0, 1.0 / m, m)
    x = np.linspace(0, 1.0 / m, m)
    interpolating_function_y = RegularGridInterpolator(
        points=(y,), values=it_r.T
    )
    interpolating_function_x = RegularGridInterpolator(
        points=(x,), values=mz_r.T
    )

    yv, xv = np.linspace(0, 1.0 / m, out_y), np.linspace(0, 1.0 / m, out_x)

    return (
        interpolating_function_x(yv),
        interpolating_function_y(xv),
    )


def filter_intensities(
    mz: np.array, it: np.array, threshold: float
) -> Tuple[np.array, np.array]:
    """
    Filter intensities by defined threshold
    :param mz: m/z
    :param it: spectrum intensities
    :param threshold: threshold to filter data
    :return: tuple with filtered data
    """
    mz = mz.copy()
    it = it.copy()
    if threshold is not None:
        above_threshold = np.where(it > threshold)
        mz = mz[above_threshold]
        it = it[above_threshold]
    return (
        mz,
        it,
    )
