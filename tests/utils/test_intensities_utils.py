import numpy as np

from msaris.utils.intensities_util import (
    get_closest_integer,
    norm,
)


def test_norm_data(data):

    test_mz, test_it = data
    max_norm = np.max(test_it) / np.sum(test_it)

    assert np.max(norm(test_it)) == max_norm


def test_close_integer():

    assert 2 == get_closest_integer(1.50001)
    assert 1 == get_closest_integer(1.49999)
