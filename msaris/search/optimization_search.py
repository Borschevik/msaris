"""
Prototype for search
Basically for now it would be harcoded for CuCl nad PdCl2 clusters
In future could would be improved and added possibility to use ANN or LP optimisation by choice
"""
from bisect import (
    bisect_left,
    bisect_right,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
from pulp import (
    LpProblem,
    LpStatus,
)

from msaris.formulas.optimisation import (
    calc_brutto_formula,
    calc_mass,
    get_coefficients,
    optimize_formula,
)
from msaris.molecule.molecule import Molecule
from msaris.utils.recognizing_utils import (
    formal_formula,
    linspace,
)


class SearchClusters:
    """
    Search isotope patterson by looking for them in set target mass
    and defined epsilong deciations
    """

    def __init__(
        self,
        mz: np.array,
        it: np.array,
        charge: int,
        *,
        threshold: float = 0.7,
        verbose: bool = False,
    ):
        self.charge = charge
        self.mz = mz
        self.it = it
        self.verbose = verbose
        self.threshold = threshold
        self.coefficients: Dict[str, int] = {}
        self.visited: List[str] = []

    @staticmethod
    def _verbose(model: LpProblem, formula: str, mass: float):
        print(f"status: {model.status}, {LpStatus[model.status]}")
        print(f"Delta m/z: {model.objective.value()}")
        print(f"Average mass = {mass}")
        print(f"Brutto formula: {formula}")

    def recognise_masses(
        self,
        target_mass: float,
        params: dict,
        *,
        epsilon_range: Tuple[int, int, float] = (
            0,
            5,
            0.25,
        ),
        calculated_ions: Optional[dict] = None,
        ions_path: Optional[str] = "./",
    ) -> list:
        """
        Method to find and calculate isotope patterns for
        defined range deviating from target mass returns list with
        parameters
        :param target_mass: target mass of isotope pattern
        :param params: parameters for otiomizaiton function
        :param epsilon_range: (start, end, step) - defining epsilon
        range with steps for calculating pattern
        :param calculated_ions: dictionary with calculated ions
        of Molecule class
        :param ions_path: folder to save calculated ions not
        present in database
        :return: list with found ions and related paramters
        """
        recognised_isotopes = []
        start, end, step = epsilon_range
        for epsilon in linspace(start, end, step):
            model = optimize_formula(
                target_mass, self.charge, epsilon, **params
            )
            model.solve()
            composition = {}
            formula = calc_brutto_formula(model)
            mass = calc_mass(model)
            if (
                LpStatus[model.status] == "Optimal"
                and formula not in self.visited
            ):
                if self.verbose:
                    self._verbose(model, formula, mass)
                for var in model.variables():
                    if var.value() != 0.0:
                        composition[f"{var.name}"[2:]] = round(
                            float(f"{var.value()}".strip())
                        )
                self.visited.append(formula)
                if calculated_ions is not None:
                    if formula in calculated_ions:
                        mol = calculated_ions[formula]
                    else:
                        mol = Molecule(formula=formula)
                        mol.calculate()
                        if ions_path:
                            mol.to_json(ions_path)
                else:
                    mol = Molecule(formula=formula)
                    mol.calculate()
                left = bisect_left(self.mz, mol.mz[0])
                right = bisect_right(self.mz, mol.mz[-1])
                spectrum = (
                    self.mz[left:right],
                    self.it[left:right],
                )
                metrics = mol.compare(spectrum)
                cosine = metrics["cosine"]
                if cosine < self.threshold:
                    formal = formal_formula(composition)
                    if self.verbose:
                        print(f"{target_mass}: {formal} {cosine}")
                    recognised_isotopes.append(
                        {
                            "formula": formula,
                            "delta": abs(mass - mol.averaged_mass),
                            "relative": (
                                max(self.it[left:right]) / max(self.it)
                            )
                            * 100,
                            "mz": mol.mz,
                            "it": mol.it,
                            "mass": mol.averaged_mass,
                            "metrics": metrics,
                            "composition": composition,
                            "spectrum": spectrum,
                        }
                    )

            self.coefficients = get_coefficients(model)

        return recognised_isotopes
