# pylint: disable=R0902,R0801
"""
Prototype for search
Basically for now it would be harcoded for CuCl nad PdCl2 clusters
In future could would be improved and added possibility to use ANN or LP optimisation by choice
"""
from collections import defaultdict
from multiprocessing import Pool
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from msaris.utils.molecule_utils import (
    color_fader,
    convert_into_formula_for_plot,
)
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
        threshold: float = 0.5,
        intensity_threshold: float = 0.01,
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
        self.intensity_threshold = intensity_threshold
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

                spectrum = (
                    self.mz.copy(),
                    self.it.copy(),
                )
                metrics = mol.compare(spectrum, windowed=True, window=0.05)
                cosine = metrics["cosine"]
                if (
                    cosine <= self.threshold
                    and metrics["relative"] > self.intensity_threshold
                ):
                    formal = formal_formula(composition)
                    if self.verbose:
                        self._verbose(model, formula, mass)
                        print(
                            f"{self.target_mass}: {formal} {cosine}\n"
                            f"Relative: {metrics['relative']}"
                        )
                    recognised.append(
                        {
                            **metrics,
                            "formula": formula,
                            "relative": metrics["relative"],
                            "mz": mol.mz,
                            "it": mol.it,
                            "mass": mol.averaged_mass,
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


# pylint: disable=R0915
def plot_results(
    mz: np.array,
    it: np.array,
    df: pd.DataFrame,
    *,
    save: bool = False,
    save_path: str = "./",
    mass_col: str = "mass",
    c1: str = "blue",
    c2: str = "red",
) -> None:
    """
    Plot results on original spectra

    :param mz: m/z of spectra
    :param it: intensity of spectra
    :param df: dataframe with data to save must contain
    :param save: save bool value to save
    :param save_path: path to save is save is true
    :param mass_col: column with mass to sort
    :param c1: the first color
    :param c2: the second color
    :return:
    """
    data_entries: Dict[str, List[Any]] = defaultdict(list)
    colors = []
    count_entries = df.shape[0]
    plt.figure(figsize=(20, 10))
    max_it = max(it)
    plt.plot(mz, (it / max_it) * 100, color="black")
    count: int = 0

    heights, locations = [], []
    for row in df.iterrows():
        colour = color_fader(c1, c2, mix=count / (count_entries + 1))
        colors.append(colour)
        ml = Molecule(formula=row[1]["brutto"])
        ml.calculate()
        brutto: str = f"{ml.brutto}"

        it_n = row[1]["relative"] * 100 / max(ml.it)
        plt.plot(
            ml.mz,
            it_n * ml.it,
            color=colour,
        )
        molecular_formula = convert_into_formula_for_plot(brutto)
        condensed_formula = convert_into_formula_for_plot(
            row[1]["brutto_formal"]
        )
        data_entries["#"].append(count + 1)
        data_entries["Mass"].append(
            round(sum(ml.mz * ml.it / np.sum(ml.it)), 2)
        )
        data_entries["Molecular formula"].append(molecular_formula)
        data_entries["Condensed formula"].append(condensed_formula)
        data_entries["Cosine"].append(f"{row[1]['cosine']:.3f}")
        data_entries["Relative int., %"].append(
            f"{row[1]['relative']*100:.3f}"
        )
        heights.append(row[1]["relative"])
        locations.append(row[1][mass_col])
        count += 1

    for ind, _ in enumerate(heights):
        x1 = [0, 0]
        y1 = [0, 0]
        if abs(locations[ind - 1] - locations[ind]) <= 10:
            x1[0] += locations[ind]
            y1[0] += heights[ind]
            heights[ind] = heights[ind] + 2
            locations[ind] = locations[ind - 1] + 15
            x1[1] += locations[ind]
            y1[1] += heights[ind] + 0.7
        plt.plot(x1, y1, "-", color=colors[ind])
        plt.text(
            locations[ind],
            heights[ind] + 1,
            f"{ind + 1}",
            color=colors[ind],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )
        count += 1

    plt.xticks(list(range(0, 1600, 200)), fontsize=18)
    plt.yticks(list(range(0, 120, 10)), fontsize=18)
    plt.xlabel("M/Z", fontsize=20)
    plt.ylabel("Intensity, %", fontsize=20)
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.autoscale(enable=True, axis="y", tight=True)
    plt.savefig(f"{save_path}_total.png", dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    df_d = pd.DataFrame(data_entries)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    fig.tight_layout()
    ax.axis("off")
    table = ax.table(
        cellText=df_d.values,
        colLabels=df_d.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    table_props = table.properties()
    table_cells = table_props["celld"]
    clr = 0
    for i, cell in enumerate(table_cells.values()):
        if i < len(table_cells) - 6:
            cell.get_text().set_fontsize(15)
            try:
                cell.get_text().set_color(colors[clr])
            except ValueError as e:
                print(e)
            if i != 0 and i % 6 == 0:
                clr += 1
    table.auto_set_column_width(col=list(range(len(df_d.columns))))

    if save:
        fig.savefig(f"{save_path}_table.png", dpi=300, bbox_inches="tight")
    plt.plot()
