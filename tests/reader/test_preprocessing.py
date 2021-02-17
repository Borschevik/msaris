import numpy as np
import pandas as pd

from msaris.reader.preprocessing import (
    filter_intensities,
    reduce_data_resolution,
    z_score,
    z_score_filtering,
)


def test_filter_intensities(data):

    mz_test, it_test = data
    mz_test, it_test = filter_intensities(mz_test, it_test, max(it_test) / 100)

    assert not np.any(it_test < max(it_test) / 100)


def test_z_score_filtering(data):

    mz_test, it_test = data
    it_filtered = z_score_filtering(
        mz_test, it_test, quantile=0.99, window=100
    )

    zscores = z_score(pd.Series(it_test), 100)
    threshold = zscores.quantile(0.99)

    assert it_test[zscores < threshold] == it_filtered


def test_data_resolution(data):

    mz_test, it_test = data

    mz_processed, it_processed = reduce_data_resolution(
        mz_test,
        it_test,
        int(mz_test.shape[0] / 10),
        int(mz_test.shape[0] / 10),
    )

    assert mz_processed.shape[0] == mz_test.shape[0] / 10
    assert it_processed.shape[0] == it_test.shape[0] / 10
