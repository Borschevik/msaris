import numpy as np


def norm(x: list) -> list:
    """
    Normalisation for intensities

    :param: x list with value to normalized
    :return: list of normalized values
    """
    return x / np.sum(x)


def get_closest_integer(value: float) -> int:
    """
    Get the closest integer to float value

    :param value: float to convert
    :return: int of value
    """
    return int(value + 0.5)