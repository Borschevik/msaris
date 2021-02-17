from msaris.reader.reader import load_data


DATA_SOURCE = "tests/resources/PdCl2_neg_000001.mzML"


def test_read_data():

    test_mz, test_it = load_data(
        DATA_SOURCE,
        range_spectrum=(0, 300),
        min_intensity=100,
    )

    assert max(test_mz) <= 300
    assert min(test_it) <= 100
