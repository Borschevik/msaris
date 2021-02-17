from pathlib import Path

import pytest

from msaris.reader.reader import load_data


BASE_DIR: Path = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data():
    """
    fixture for getting test data
    :return: test data
    """
    DATA_SOURCE = f"{BASE_DIR}/tests/resources/PdCl2_neg_000001.mzML"
    test_mz, test_it = load_data(
        DATA_SOURCE,
        range_spectrum=(0, 300),
        min_intensity=100,
    )
    return (test_mz, test_it)
