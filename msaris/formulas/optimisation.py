"""

    Prototype for pattern calculation in future realeases would be expanded beyond CuCl and PdCl2 case

"""
# TODO: prototype need to find solution for general case
from pulp import (
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
)

from msaris.formulas.monoisotopic_masses import *


def optimize_formula(
    target_mass: float,
    charge: int,
    epsilon: float,
    *,
    no_TBA: bool = True,
    no_K: bool = True,
    no_MeOH: bool = True,
    no_Cu: bool = False,
    no_Pd1: bool = True,
    no_Pd: bool = False,
    no_NaTFA: bool = True,
    no_OH: bool = True,
    no_H2O: bool = True,
    no_O2: bool = True,
    no_O: bool = True,
    no_N2: bool = True,
    no_Na: bool = True,
    no_CH3CN: bool = True,
    no_Br: bool = True,
) -> LpProblem:
    """
    Pattern calculation via linear optimization

    :param target_mass: target mass float
    :param charge: isotope pattern charge
    :param epsilon: allowed deviation from mass
    :param no_TBA:
    :param no_K:
    :param no_Br
    :param no_MeOH:
    :param no_Cu:
    :param no_Pd:
    :param no_NaTFA:
    :param no_OH:
    :param no_H2O:
    :param no_O2:
    :param no_O:
    :param no_N2:
    :param no_Na:
    :param no_CH3CN:
    :return: model with calculated coefficients
    """

    # Create the model
    model = LpProblem(name="find-formula", sense=LpMinimize)

    # Transition metals
    n_Pd1 = LpVariable(name="n_Pd1", lowBound=0, cat="Integer")
    n_Pd2 = LpVariable(name="n_Pd2", lowBound=0, cat="Integer")
    n_Cu1 = LpVariable(name="n_Cu1", lowBound=0, cat="Integer")
    n_Cu2 = LpVariable(name="n_Cu2", lowBound=0, cat="Integer")

    # Main-group cations
    n_Na = LpVariable(name="n_Na", lowBound=0, cat="Integer")
    n_K = LpVariable(name="n_K", lowBound=0, cat="Integer")
    n_TBA = LpVariable(name="n_TBA", lowBound=0, cat="Integer")

    # Anions
    n_OH = LpVariable(name="n_OH", lowBound=0, cat="Integer")
    n_Cl = LpVariable(name="n_Cl", lowBound=0, cat="Integer")
    n_Br = LpVariable(name="n_Br", lowBound=0, cat="Integer")
    n_CF3COO = LpVariable(name="n_CF3COO", lowBound=0, cat="Integer")
    n_O2_1 = LpVariable(name="n_O2_1", lowBound=0, cat="Integer")
    n_O2_2 = LpVariable(name="n_O2_2", lowBound=0, cat="Integer")
    n_O = LpVariable(name="n_O", lowBound=0, cat="Integer")

    # Ligands
    n_CH3CN = LpVariable(name="n_CH3CN", lowBound=0, cat="Integer")
    n_CH3OH = LpVariable(name="n_CH3OH", lowBound=0, cat="Integer")
    n_N2 = LpVariable(name="n_N2", lowBound=0, cat="Integer")
    n_H2O = LpVariable(name="n_H2O", lowBound=0, cat="Integer")
    # n_O2

    # Add the constraints to the model
    model += (
        n_Pd1
        + 2 * n_Pd2
        + n_Cu1
        + 2 * n_Cu2
        + n_Na
        + n_K
        + n_TBA
        - n_OH
        - n_Cl
        - n_Br
        - n_CF3COO
        - 2 * n_O2_2
        - n_O2_1
        - 2 * n_O
        == charge,
        "Charge_constraint",
    )
    model += (
        4 * n_Pd1 + 4 * n_Pd2 + 4 * n_Cu1 + 4 * n_Cu2
        >= n_OH
        + n_Cl
        + n_Br
        + n_CF3COO
        + n_CH3CN
        + n_CH3OH
        + n_N2
        + n_H2O
        + 2 * n_O2_2
        + n_O2_1
        + 2 * n_O,
        "Valence_constraint",
    )
    model += (
        n_Pd1 + n_Pd2 + n_Cu1 + n_Cu2 >= n_Na + n_K + n_TBA,
        "Impurity_constraint: cations",
    )
    model += (n_Cl >= 2 * n_Br + 2 * n_OH, "Impurity_constraint: anions")
    model += (n_N2 <= 2, "Impurity_constraint: N2 activation")
    model += (n_H2O <= 1, "Impurity_constraint: H2O activation")
    model += (n_CH3OH <= 3, "Impurity_constraint: MeOH activation")
    model += (
        n_CH3CN >= n_CH3OH + n_H2O + n_CF3COO + n_N2,
        "Acetonitrile constraint",
    )
    model += (n_O2_2 + n_O2_1 <= 1, "Impurity_constraint: O2 activation")
    model += (n_O <= 1, "Impurity_constraint: oxidation")
    model += (n_Na <= 1, "Impurity_constraint: Natrium")
    model += (n_Cu1 - n_Cu2 >= 0, "Oxidation_constraint: Cu")

    model += (
        n_Pd1 * m_Pd
        + n_Pd2 * m_Pd
        + n_Cu1 * m_Cu
        + n_Cu2 * m_Cu
        + n_Na * m_Na
        + n_K * m_K
        + n_TBA * m_TBA
        + n_OH * m_OH
        + n_Cl * m_Cl
        + n_Br * m_Br
        + n_CF3COO * m_CF3COO
        + n_CH3CN * m_CH3CN
        + n_CH3OH * m_CH3OH
        + n_N2 * m_N2
        + n_H2O * m_H2O
        + n_O2_1 * m_O2
        + n_O2_2 * m_O2
        + n_O * m_O
        - target_mass
        <= epsilon,
        "Epsilon_right",
    )

    model += (
        n_Pd1 * m_Pd
        + n_Pd2 * m_Pd
        + n_Cu1 * m_Cu
        + n_Cu2 * m_Cu
        + n_Na * m_Na
        + n_K * m_K
        + n_TBA * m_TBA
        + n_OH * m_OH
        + n_Cl * m_Cl
        + n_Br * m_Br
        + n_CF3COO * m_CF3COO
        + n_CH3CN * m_CH3CN
        + n_CH3OH * m_CH3OH
        + n_N2 * m_N2
        + n_H2O * m_H2O
        + n_O2_1 * m_O2
        + n_O2_2 * m_O2
        + n_O * m_O
        - target_mass
        >= -epsilon,
        "Epsilon_left",
    )

    # Control element composition
    if no_Br:
        model += (n_Br == 0, "Without TBA: Br-")
    if no_TBA:
        model += (n_TBA == 0, "Without TBA: TBA+")
    if no_K:
        model += (n_K == 0, "Without K+")
    if no_MeOH:
        model += (n_CH3OH == 0, "Without MeOH")
    if no_CH3CN:
        model += (n_CH3CN == 0, "Without MeCN")
    if no_Cu:
        model += (n_Cu1 == 0, "Without Cu(I)")
        model += (n_Cu2 == 0, "Without Cu(II)")
    if no_Pd:
        model += (n_Pd1 == 0, "Without Pd(I)")
        model += (n_Pd2 == 0, "Without Pd(II)")
    if no_NaTFA:
        if no_Na:
            model += (n_Na == 0, "Without Na+")
        model += (n_CF3COO == 0, "Without TFA-")
    if no_OH:
        model += (n_OH == 0, "Without OH-")
    if no_H2O:
        model += (n_H2O == 0, "Without H2O")
    if no_N2:
        model += (n_N2 == 0, "Without N2")
    if no_O2:
        model += (n_O2_1 == 0, "Without superoxide O2")
        model += (n_O2_2 == 0, "Without peroxide O2")
    if no_O:
        model += (n_O == 0, "Neglect oxidation by atomic O")

    # Add the objective function to the model
    model += lpSum(
        [
            n_Pd1 * m_Pd,
            n_Pd2 * m_Pd,
            n_Cu1 * m_Cu,
            n_Cu2 * m_Cu,
            n_Na * m_Na,
            n_K * m_K,
            n_TBA * m_TBA,
            n_OH * m_OH,
            n_Cl * m_Cl,
            n_Br * m_Br,
            n_CF3COO * m_CF3COO,
            n_CH3CN * m_CH3CN,
            n_CH3OH * m_CH3OH,
            n_N2 * m_N2,
            n_O2_1 * m_O2,
            n_O2_2 * m_O2,
            n_O * m_O,
            n_H2O * m_H2O,
            -target_mass,
        ]
    )

    return model


def get_coefficients(model: LpProblem) -> dict:
    """
    Retrieves from calculated mode coefficients in dictionary format

    :param model: calculated model
    :return: coefficients in dictionary format
    """
    variables: dict = {}
    for var in model.variables():
        variables[f"{var.name}"[2:]] = int(
            round(float(f"{var.value()}".strip()))
        )
    return variables


def calc_brutto_formula(model: LpProblem) -> str:
    """
    Get brutto formula for calculated model

    :param model: calculated model
    :return: brutto formula in string format
    """

    brutto_formula = []
    brutto_out = ""
    d_brutto = {
        "Pd": 0,
        "Cu": 0,
        "K": 0,
        "Na": 0,
        "Br": 0,
        "Cl": 0,
        "C": 0,
        "N": 0,
        "O": 0,
        "F": 0,
        "H": 0,
    }
    d_units = get_coefficients(model)
    if "Pd1" in d_units:
        d_brutto["Pd"] += d_units["Pd1"]
    if "Pd2" in d_units:
        d_brutto["Pd"] += d_units["Pd2"]

    if "Cu1" in d_units:
        d_brutto["Cu"] += d_units["Cu1"]
    if "Cu2" in d_units:
        d_brutto["Cu"] += d_units["Cu2"]

    if "Na" in d_units:
        d_brutto["Na"] += d_units["Na"]

    if "K" in d_units:
        d_brutto["K"] += d_units["K"]

    if "O" in d_units:
        d_brutto["O"] += d_units["O"]

    if "CF3COO" in d_units:
        d_brutto["C"] += 2 * d_units["CF3COO"]
        d_brutto["F"] += 3 * d_units["CF3COO"]
        d_brutto["O"] += 2 * d_units["CF3COO"]

    if "TBA" in d_units:
        d_brutto["C"] += 16 * d_units["TBA"]
        d_brutto["H"] += 36 * d_units["TBA"]
        d_brutto["N"] += 1 * d_units["TBA"]

    if "OH" in d_units:
        d_brutto["O"] += 1 * d_units["OH"]
        d_brutto["H"] += 1 * d_units["OH"]

    if "Cl" in d_units:
        d_brutto["Cl"] += d_units["Cl"]

    if "Br" in d_units:
        d_brutto["Br"] += d_units["Br"]

    if "CH3CN" in d_units:
        d_brutto["C"] += 2 * d_units["CH3CN"]
        d_brutto["H"] += 3 * d_units["CH3CN"]
        d_brutto["N"] += 1 * d_units["CH3CN"]

    if "CH3OH" in d_units:
        d_brutto["C"] += 1 * d_units["CH3OH"]
        d_brutto["H"] += 4 * d_units["CH3OH"]
        d_brutto["O"] += 1 * d_units["CH3OH"]

    if "H2O" in d_units:
        d_brutto["O"] += 1 * d_units["H2O"]
        d_brutto["H"] += 2 * d_units["H2O"]

    if "N2" in d_units:
        d_brutto["N"] += 2 * d_units["N2"]

    if "O2_1" in d_units:
        d_brutto["O"] += 2 * d_units["O2_1"]
    if "O2_2" in d_units:
        d_brutto["O"] += 2 * d_units["O2_2"]

    for element in d_brutto:
        if d_brutto[element] > 0:
            brutto_formula += [element + str(d_brutto[element])]

    brutto_formula = sorted(brutto_formula)

    brutto_out = brutto_out.join(brutto_formula)

    return brutto_out


def calc_mass(model: LpProblem) -> float:
    """
    Function to get isotope mass for calculated pattern

    :param model: pulp model
    :return: calculated isotope mass
    """

    ion_mass = 0.0

    for var in model.variables():
        if var.value() != 0.0:
            ion_mass += d_m["m_" + f"{var.name}"[2:]] * float(
                f"{var.value()}".strip()
            )
    return ion_mass
