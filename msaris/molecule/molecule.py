"""
Molecule generation and representation for generating theoretical spectre
"""
import json
import os
from bisect import (
    bisect_left,
    bisect_right,
)
from typing import (
    List,
    Optional,
)

import IsoSpecPy as iso
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer  # pylint: disable=C0411
from molmass import Formula
from scipy.interpolate import interp1d
from scipy.spatial import (
    ckdtree,
    distance,
)

from msaris.utils.distributions_util import generate_gauss_distribution
from msaris.utils.intensities_util import (
    get_spectrum_by_close_values,
    norm,
)
from msaris.utils.molecule_utils import convert_into_formula_for_plot


class Molecule:  # pylint: disable=R0902
    """
    Molecule generation and saving for performing molecule search
    """

    def __init__(self, *, formula: str = "", delta: float = 0.5):
        self.formula = formula  # saving for using to refer
        self.brutto = self._get_brutto() if formula else None
        self.mass_out: List = []
        self.intens_out: List = []
        self.mz: np.array = np.array([])
        self.it: np.array = np.array([])
        self.bar_raw = np.array([])
        self.averaged_mass: float = 0.0
        self.delta: float = delta

    def _get_brutto(self) -> str:
        """
        Generating brutto formula from provided one

        :returns: brutto formula
        """
        f = Formula(self.formula)
        return "".join(map(lambda x: f"{x[0]}{x[1]}", f.composition()))

    def _get_bar_isotope(self):
        """
        Generating raw bar isotope pattern
        :return:
        """
        non_zero = np.argwhere(np.array(self.intens_out) > 0)
        self.bar_raw = self.intens_out.copy()
        for ind in non_zero:
            left = bisect_left(self.mass_out, self.mass_out[ind] - self.delta)
            right = bisect_right(
                self.mass_out, self.mass_out[ind] + self.delta
            )
            self.bar_raw[left:right] = self.intens_out[ind]

    def select_windowed_signals_by_molecule(
        self, mz: np.array, it: np.array, *, window: float = 0.1
    ):
        """
        Select from spectra peaks around generated theoretical isotope pattern

        :param mz: m/z of mass spectra
        :param it: intensity of mass spectra
        :param window: window for selecting around theoretical isotope
        :return:
        """
        bar_mz: List = list()
        bar_it: List = list()
        new_mz, new_it = mz.copy(), np.zeros(len(it))
        visited: List = list()
        indexes: List = []
        tree = ckdtree.cKDTree(np.array([self.mz, self.mz]).T)
        for _, val in enumerate(self.mass_out):
            inds = tree.query_ball_point((val, val), window)
            S1, S2 = set(inds), set(visited)
            if S1.intersection(S2):
                continue
            visited.extend(inds)
            bar_mz.append(np.mean(self.mz[inds]))
            bar_it.append(np.mean(self.it[inds]))

        tree = ckdtree.cKDTree(np.array([mz, mz]).T)
        for mz_t_i in bar_mz:
            inds = tree.query_ball_point(
                (mz_t_i, mz_t_i),
                window,
                eps=0.0,
            )
            indexes.extend(inds)

        indexes = list(set(indexes))
        new_mz[indexes] = mz[indexes]
        new_it[indexes] = it[indexes]

        return new_mz, new_it, bar_mz, bar_it

    def calculate(
        self, *, resolution: int = 20, ppm: int = 50, scale: bool = False
    ) -> None:

        """
        :param resolution: generating m/z and intensities for provided formula
        :return: None
        """
        scaling: np.array

        try:
            sp = iso.IsoTotalProb(formula=self.brutto, prob_to_cover=0.99999)
        except ValueError:
            raise ValueError(f"Invalid {self.formula}")

        for mass, prob in sp:
            prob *= 100.0
            self.mass_out += [mass]
            self.intens_out += [prob]

        self.mz, self.it, self.averaged_mass = generate_gauss_distribution(
            self.mass_out, self.intens_out, ppm=ppm, resolution=resolution
        )

        if scale:
            scaling = 100 / max(self.it)
        else:
            scaling = max(self.intens_out) / max(self.it)
        # scaling resulting curve
        self.it = self.it * scaling

    def plot(
        self,
        *,
        save: bool = False,
        path: str = "./",
        name: Optional[str] = None,
    ) -> None:
        """
        Plot spectra

        :param save: bool value to save image of spectra
        :param path: path to save image
        :param name: name format

        :return: None
        """
        # TODO: change to be more flexible for output params
        plt.rcParams["figure.figsize"] = (30, 30)
        # plot settings
        fig, (ax_spiketrain, ax_filtered) = plt.subplots(2, 1, sharex=True)
        ax_spiketrain.tick_params(axis="x", labelbottom=True, rotation=-90)
        ax_spiketrain.tick_params(axis="both")
        # tick parameters
        plt.xticks(
            np.arange(
                int(min(self.mass_out)) - 1, int(max(self.mass_out)) + 2, 1.0
            ),
            rotation=-90,
        )
        markerline, stemlines, baseline = ax_spiketrain.stem(
            self.mass_out,
            self.intens_out,
            use_line_collection="True",
            linefmt="grey",
            markerfmt="D",
            basefmt="k-",
            bottom=0,
        )
        markerline.set_markerfacecolor("none")
        plt.setp(stemlines, "linewidth", 0.9)
        plt.setp(markerline, "linewidth", 0.8)
        plt.setp(baseline, "linewidth", 0.9)
        ax_spiketrain.set_title("Original spike train from IsoSpec data")
        ax_spiketrain.set_ylabel("Relative intensity, %")
        ax_spiketrain.set_xlabel("Mass, Da")

        ax_filtered.plot(self.mz, self.it, color="blue", lw=1.2)
        # axes labels
        ax_filtered.set_title("Gaussian-filtered predicted spectra")
        ax_filtered.set_ylabel("Relative intensity, %")
        ax_filtered.set_xlabel("Mass, Da")
        plt.rcParams.update({"font.size": 30})

        if save:
            name = f"{path}{name}.png" if name else f"{path}{self.formula}.png"
            fig.savefig(name, dpi=300, format="png", bbox_inches="tight")

        plt.show()
        plt.close()

    def to_dict(self) -> dict:
        """
        Present result in dict format
        :return: dictionary of the main parameters
        """
        return {
            "formula": self.formula,
            "brutto": self.brutto,
            "mz": self.mz.tolist(),
            "it": self.it.tolist(),
            "mass_out": self.mass_out,
            "intens_out": self.intens_out,
            "averaged_mass": self.averaged_mass,
        }

    def to_json(self, path: str = "./", name: Optional[str] = None) -> None:
        """
        Saves the molecule's to json

        :param path: string default save to place where executed
        :param name: redifine name default is formula with .mol format
        :return: None
        """

        if not os.path.isdir(path):
            os.makedirs(path)

        name = f"{self.formula}.json" if name is None else f"{name}.json"
        if not path.endswith("/"):
            path = f"{path}/"

        with open(f"{path}{name}", "w") as outfile:
            json.dump(self.to_dict(), outfile)

        typer.echo(f"✨ JSON with was created: {os.path.abspath(path)}{name} ✨")

    def read_dict_data(self, data: dict) -> None:
        """
        Gets Molecule from dictionary representation of molecule

        :param data: data in dictionary format
        :return: None
        """
        for field, value in data.items():
            if field in ("mz", "it"):
                value = np.array(value)
            setattr(self, field, value)

    def load(self, file_path: str) -> None:
        """
        Load file in JSON format

        :param: Path to load data
        :return: None
        """
        with open(file_path, "r") as file:
            self.read_dict_data(json.load(file))

    def __str__(self) -> str:
        return self.formula

    def __repr__(self) -> str:
        return (
            f"<Molecule(formula={self.formula},"
            f" weighted_mass={self.averaged_mass})>"
        )

    def _get_delta_avg(self, spectrum: tuple) -> float:
        mz_av, it_av = spectrum
        it_av = it_av / sum(it_av)
        return abs(sum(mz_av * it_av) - self.averaged_mass)

    def compare(
        self,
        experimental: tuple,
        *,
        windowed: bool = True,
        window: float = 0.1,
    ) -> dict:

        """
        Function to perform calculations for the theoretical and experimental spectrum
        Based on interpolation selected peaks are recalculated to the same mz_t value

        :param experimental: m/z and it of experimental data
        :param windowed: define window to cut around from experimental spectrum by using theoretical
        :param window: window to cut around experimental peaks

        :return: calculated metrics for the selected spectra
        """
        metrics: dict = {}
        mz_t, it_t = self.mz.copy(), self.it.copy()
        mz_e, it_e = experimental
        max_spectrum = max(it_e)
        if windowed:
            mz_e, it_e, _, _ = self.select_windowed_signals_by_molecule(
                mz_e, it_e, window=window
            )
        max_it_x_ind, max_it_t_ind = np.argmax(it_e), np.argmax(self.it)
        mz_max_x, mol_mz = (
            mz_e[max_it_x_ind],
            self.mz[max_it_t_ind],
        )
        metrics["relative"] = max(it_e) / max_spectrum
        metrics["delta_max"] = abs(mz_max_x - mol_mz)

        interpol_t = interp1d(
            mz_t, norm(it_t), bounds_error=False, fill_value=(0, 0)
        )
        interpol_e = interp1d(
            mz_e, norm(it_e), bounds_error=False, fill_value=(0, 0)
        )
        theory = interpol_t(mz_e) * 100
        exp = interpol_e(mz_e) * 100

        metrics["cosine"] = distance.cosine(theory, exp)
        metrics["delta_avg"] = self._get_delta_avg((mz_e, it_e))
        # TODO: improve and add other statistics calculations
        return metrics


def compare_and_visualize(
    mz: np.array,
    it: np.array,
    formula: str,
    *,
    save: bool = False,
    path: str = "./",
    window: float = 0.1,
    font: int = 18,
):
    """
    Comparing and visualize and isotope pattern with mass specter

    :param mz: m/z
    :param it: intensity
    :param formula: formula of isotope pattern in string format
    :param save: bool value to save plot
    :param path: path where save plot
    :param window: parameter to decide bar plot length and
    :param font: font size in int
    :return:
    """
    mol = Molecule(formula=formula)
    mol.calculate()
    mz, it = mz.copy(), it.copy()

    mz_f, it_f, _, _ = get_spectrum_by_close_values(
        mz, it, mol.mz[0], mol.mz[-1]
    )
    mpl.rcParams["xtick.labelsize"] = font
    mpl.rcParams["ytick.labelsize"] = font
    mz_c, it_c, bar_mz, bar_it = mol.select_windowed_signals_by_molecule(
        mz_f, it_f, window=window
    )
    max_it_x_ind, max_it_t_ind = np.argmax(it_c), np.argmax(mol.it)
    mz_max_x, it_max_x, mol_mz_x, _ = (
        mz_c[max_it_x_ind],
        it_c[max_it_x_ind],
        mol.mz[max_it_t_ind],
        mol.it[max_it_t_ind],
    )
    metrics = mol.compare(
        (
            mz_c,
            it_c,
        )
    )
    stats = {
        "delta": abs(mz_max_x - mol_mz_x),
        "cosine": metrics["cosine"],
        "relative": it_max_x / max(it),
    }
    _, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.bar(
        bar_mz,
        (bar_it / max(bar_it)) * 100,
        width=window,
        align="center",
        alpha=1,
        color="r",
    )
    ind = np.argsort(mz_c)
    ax.plot(
        mz_c[ind],
        (it_c[ind] / it_max_x) * 100,
        color="black",
    )
    plotted_formula = convert_into_formula_for_plot(formula)
    ax.set_xlabel("M/Z", fontsize=20)
    ax.set_ylabel("Intensity", fontsize=20)
    labels = [
        plotted_formula,
        f"Delta m/z: {stats['delta']:.3f}",
        f"Cosine: {stats['cosine']:.3f}",
        f"Relative: {stats['relative']:.3f}",
    ]
    ax.text(
        0.8,
        0.8,
        "\n".join(labels),
        color="black",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=font,
        transform=ax.transAxes,
    )
    ax.set_title(f"{plotted_formula}", fontsize=20)
    if save and path:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/{formula}.png", dpi=600)
    plt.show()
    plt.close()
