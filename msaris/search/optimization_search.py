"""
Prototype for search
Basically for now it would be harcoded for CuCl nad PdCl2 clusters
In future could would be improved and added possibility to use ANN or LP optimisation by choice
"""
from bisect import bisect_left, bisect_right
from typing import Optional, Tuple

import numpy as np
from pulp import LpStatus

from msaris.formulas.optimisation import (
    optimize_formula,
    calc_mass,
    calc_brutto_formula,
    get_coefficients
)
from msaris.molecule.molecule import Molecule
from msaris.utils.recognizing_utils import formal_formula, linspace


class SearchClusters:

    def __init__(
            self,
            mz: np.array,
            it: np.array,
            charge: int,
            *,
            threshold: float = 0.7
            ):
        self.charge = charge
        self.mz = mz
        self.it = it
        self.threshold = threshold
        self.coefficients = {}
        self.visited = []

    def recognise_masses(
            self,
            target_mass: float,
            params: dict,
            *,
            epsilon_range: Tuple[int, int, float] = (0, 5, 0.25,),
            calculated_ions: Optional[dict] = None,
            ions_path: Optional[str] = "./"
        ) -> list:
        recognised_isotopes = []
        start, end, step = epsilon_range
        for epsilon in linspace(start, end, step):
            model = optimize_formula(
                        target_mass,
                        self.charge,
                        epsilon,
                        **params
            )
            model.solve()
            composition = {}
            formula = calc_brutto_formula(model)
            mass = calc_mass(model)
            if LpStatus[model.status] == "Optimal" and formula not in self.visited:
                print(f"status: {model.status}, {LpStatus[model.status]}")
                print(f"Delta m/z: {model.objective.value()}")
                print(f"Average mass = {mass}")
                print(f"Brutto formula: {formula}")
                for var in model.variables():
                    if var.value() != 0.0:
                        composition[f"{var.name}"[2:]] = round(float(f"{var.value()}".strip()))
                self.visited.append(formula)
                if calculated_ions is not None:
                    if formula in calculated_ions:
                        mol = calculated_ions[formula]
                    else:
                        mol = Molecule(formula)
                        mol.calculate()
                        if ions_path:
                            mol.to_json(ions_path)
                else:
                    mol = Molecule(formula)
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
                    print(f"{target_mass}: {formal} {cosine}")
                    recognised_isotopes.append(
                        {
                            "formula": formal,
                            "delta": abs(mass - mol.averaged_mass),
                            "relative": (
                                                max(self.it[left:right]) / max(self.it)
                                        ) * 100,
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
