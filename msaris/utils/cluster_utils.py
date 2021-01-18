from typing import Iterable, TypeVar

import numpy as np
from scipy.spatial import ckdtree

IND = TypeVar('IND', Iterable, np.ndarray, int, float)


def find_indexes(X: np.ndarray, val: float, max_d: float) -> IND:
    """
    Find indexes close to provided value on predefined distance

    :param X: list of values to perform search
    :param val: float of value indexes to find
    :param max_d: max radius to search for values
    :return: list of indexes
    """
    tree = ckdtree.cKDTree(np.array([X, X]).T)

    return tree.query_ball_point((val, val,), max_d, eps=0.0)
