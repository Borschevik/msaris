"""
    Function for performing transformations of the intensities
"""
import numpy as np


def norm(intensities: list) -> list:
    """
    Normalisation for intensities

    :param: x list with value to normalized
    :return: list of normalized values
    """
    return intensities / np.sum(intensities)


def get_closest_integer(value: float) -> int:
    """
    Get the closest integer to float value

    :param value: float to convert
    :return: int of value
    """
    return int(value + 0.5)
