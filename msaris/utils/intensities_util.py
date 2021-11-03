"""
    Function for performing transformations of the intensities
"""
from bisect import (
    bisect_left,
    bisect_right,
)
from typing import Tuple

import numpy as np


def norm(intensities: list) -> list:
    """
    Normalisation for intensities

    :param: x list with value to normalized
    :return: list of normalized values
    """
    intensities_c = intensities.copy()
    return intensities_c / np.sum(intensities_c)


def get_closest_integer(value: float) -> int:
    """
    Get the closest integer to float value

    :param value: float to convert
    :return: int of value
    """
    return int(value + 0.5)


def get_spectrum_by_close_values(
    mz: list,
    it: list,
    left_border: float,
    right_border: float,
    *,
    eps: float = 0.0
) -> Tuple[list, list, int, int]:
    """
    Function to get segment of spectrum by left and right
    border
    :param mz: m/z array
    :param it: it intensities
    :param left_border: left border
    :param right_border: right border
    :param eps: epsilon to provide regulation of borders
    :return: closest to left and right border values of spectrum, left and right
    """
    mz, it = mz.copy(), it.copy()
    left = bisect_left(mz, left_border - eps)
    right = bisect_right(mz, right_border + eps)

    return mz[left:right].copy(), it[left:right].copy(), left, right


def get_indexes(mz, peak, *, eps=0) -> Tuple[list, int, int]:
    """
    Return indexes close to value
    :param mz: m/z array
    :param peak: peak
    :param eps: window from peak to search
    :return: indexes, left and right border
    """
    left = bisect_left(mz, peak - eps)
    right = bisect_right(mz, peak + eps)
    return list(range(left, right)), left, right
