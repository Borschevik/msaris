import re
from collections import defaultdict

import numpy as np


def formal_formula(composition: dict) -> str:
    """
    Generate formal formula to have better text representation of the brutto formula

    :param composition: dict with defined fomula variable from optimal calculations

    :returns: str
    """
    metals = ["Cu", "Pd", "K", "Na"]
    anions = ["Cl", "Br"]
    metal = defaultdict(int)
    anion = defaultdict(int)
    ligands = ""
    for group, number in composition.items():
        without_num = re.split("\d", group)[0]
        if without_num in metals:
            metal[without_num] += number
        elif without_num in anions:
            anion[without_num] += number
        else:
            ligands = f"{ligands}({group}){number}"
    molecule = "".join([f"{k}{v}" for k, v in metal.items()])
    molecule += "".join([f"{k}{v}" for k, v in anion.items()])
    molecule += ligands
    return molecule


def linspace(start: float, stop: float, step: float=1.) -> np.array:
    """
    Custom linspace to perform generation of array stepwise

    :param start: start value
    :param stop: end value
    :param step: step

    :return: array of value
    """

    return np.linspace(start, stop, int((stop - start) / step + 1))

