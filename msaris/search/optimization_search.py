# pylint: disable=R0902,R0801
"""
Prototype for search
Basically for now it would be harcoded for CuCl nad PdCl2 clusters
In future could would be improved and added possibility to use ANN or LP optimisation by choice
"""
from multiprocessing import Pool
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
from msaris.utils.intensities_util import get_spectrum_by_close_values
from msaris.utils.recognizing_utils import (
    formal_formula,
    linspace,
)


def calculate_formula(
    target_mass: float, charge: int, epsilon: float, params: dict
) -> tuple:
    """
    Calculating formula for selected target mass and charge
    :param target_mass: float target pattern mass
    :param charge: target pattern charge
    :param epsilon: float parameter max deviation from target mass
    :param params: list of params for model
    :return: model with established formula and mass
    """
    model = optimize_formula(target_mass, charge, epsilon, **params)
    model.solve()
    formula = calc_brutto_formula(model)
    mass = calc_mass(model)
    return (
        model,
        formula,
        mass,
    )


class SearchClusters:
    """
    Search isotope patterson by looking
    for them in set target mass
    and defined epsilon definition
    """

    def __init__(
        self,
        mz: np.array,
        it: np.array,
        charge: int,
        *,
        threshold: float = 0.7,
        verbose: bool = False,
        adjusted: bool = False,
        njobs: int = 1,
    ):
        """
        Initializing optimization search algorithm search is based to find list with metrics for
        Optimal molecule formula which would satisfy threshold for defined metric
        :param mz: original spectrum m/z, used to calculate metric
        :param it: original spectrum intensities
        :param charge: charge of found ions
        :param threshold: threshold for metric
        :param adjusted: adjust m/z values according to
        delta between theoretical and experimental values
        :param verbose: parameter to show logs for calculating isotope pattern formulas
        :param njobs: parameter for using number of CPU for calculating jobs in parallel
        """
        self.charge = charge
        self.mz = mz
        self.it = it
        self.verbose = verbose
        self.threshold = threshold
        self.coefficients: Dict[str, int] = {}
        self.visited: List[str] = []
        self.adjusted = adjusted
        self.njobs = njobs
        self.params: dict = {}
        self.target_mass: float = 0.0
        self.calculated_ions: Optional[dict] = None
        self.ions_path: Optional[str] = "./"

    @staticmethod
    def _verbose(model: LpProblem, formula: str, mass: float):
        print(f"status: {model.status}, {LpStatus[model.status]}")
        print(f"Delta m/z: {model.objective.value()}")
        print(f"Average mass = {mass}")
        print(f"Brutto formula: {formula}")
        report = ",\n".join(
            ": ".join((key, str(val)))
            for (key, val) in get_coefficients(model).items()
        )
        print(f"Composition: \n{report}")

    @staticmethod
    def _get_delta_avg(mol: Molecule, spectrum: list) -> float:
        mz_av, it_av = spectrum
        it_av = it_av / sum(it_av)
        return abs(sum(mz_av * it_av) - mol.averaged_mass)

    def __calculate_results(self, epsilon_range: tuple) -> list:
        recognised = []
        start, end, step = epsilon_range
        pool = Pool(processes=self.njobs)
        params_ = []
        for eps in linspace(start, end, step):
            params_.append((self.target_mass, self.charge, eps, self.params))
        results = pool.starmap(calculate_formula, params_)
        for result in results:
            model, formula, mass = result
            if (
                LpStatus[model.status] == "Optimal"
                and formula not in self.visited
            ):
                composition = get_coefficients(model)
                self.visited.append(formula)
                if (
                    self.calculated_ions is not None
                    and formula in self.calculated_ions
                ):
                    mol = self.calculated_ions[formula]
                else:
                    mol = Molecule(formula=formula)
                    mol.calculate()
                    if self.ions_path:
                        mol.to_json(self.ions_path)
                mz_f, it_f, _, _ = get_spectrum_by_close_values(
                    self.mz, self.it, mol.mz[0], mol.mz[-1]
                )
                mz_f, it_f = mz_f.copy(), it_f.copy()
                m_x = mz_f[np.argmax(it_f)]
                m_t = mol.mz[np.argmax(mol.it)]
                delta_b = m_x - m_t

                spectrum = [
                    mz_f,
                    it_f,
                ]
                delta_avg = self._get_delta_avg(mol, spectrum)
                metrics = mol.compare(spectrum)
                cosine = metrics["cosine"]
                if cosine <= self.threshold:
                    formal = formal_formula(composition)
                    if self.verbose:
                        self._verbose(model, formula, mass)
                        print(f"{self.target_mass}: {formal} {cosine}")
                    recognised.append(
                        {
                            "formula": formula,
                            "delta_max": abs(delta_b),
                            "delta_avg": delta_avg,
                            "relative": (max(it_f) / max(self.it)) * 100,
                            "mz": mol.mz,
                            "it": mol.it,
                            "mass": mol.averaged_mass,
                            "metrics": metrics,
                            "composition": composition,
                            "spectrum": spectrum,
                        }
                    )

        return recognised

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
        :param params: parameters for optimization function
        :param epsilon_range: (start, end, step) - defining epsilon
        range with steps for calculating pattern
        :param calculated_ions: dictionary with calculated ions
        of Molecule class
        :param ions_path: folder to save calculated ions not
        present in database
        :return: list with found ions and related parameters
        """
        self.params = params
        self.target_mass = target_mass
        self.calculated_ions = calculated_ions
        self.ions_path = ions_path
        return self.__calculate_results(epsilon_range)
