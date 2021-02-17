from msaris.utils.recognizing_utils import (
    formal_formula,
    linspace,
)


def test_linspace():
    values = list(linspace(0, 5, 1.0))
    assert values == [0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_formal_formula():

    composition = {"Cu": 1, "Cl": 3, "Na": 2, "CH3CN": 1}

    formula = formal_formula(composition)

    assert formula == "Cu1Na2Cl3(CH3CN)1"
