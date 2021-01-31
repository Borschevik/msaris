"""
Molecule generation and representation for generating theoretical spectre
"""
import json
import os
from typing import (
    List,
    Optional,
)

import IsoSpecPy as iso
import matplotlib.pyplot as plt
import numpy as np
import typer
from molmass import Formula
from scipy.interpolate import interp1d
from scipy.spatial import distance

from msaris.utils.distributions_util import generate_gauss_distribution
from msaris.utils.intensities_util import norm


class Molecule:
    """
    Molecule generation and saving for performing molecule search
    """

    def __init__(self, *, formula: str = ""):
        self.formula = formula  # saving for using to refer
        self.brutto = self._get_brutto() if formula else None
        self.mass_out: List = []
        self.intens_out: List = []
        self.mz, self.it, self.averaged_mass = np.array([]), np.array([]), 0.0

    def _get_brutto(self) -> str:
        """
        Generating brutto formula from provided one

        :returns: brutto formula
        """
        f = Formula(self.formula)
        return "".join(map(lambda x: f"{x[0]}{x[1]}", f.composition()))

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

    def compare(self, experimental: tuple) -> dict:

        """
        Function to perform calculations for the theoretical and experimental spectrum
        Based on interpolation selected peaks are recalculated to the same mz_t value

        :param experimental: m/z and it of experimantal data

        :return: calculated metrics for the selected spectras
        """
        metrics: dict = {}
        mz_t, it_t = self.mz.copy(), self.it.copy()
        mz_e, it_e = experimental
        it_t = norm(it_t)
        it_e = norm(it_e)

        interpol_t = interp1d(
            mz_t, norm(it_t), bounds_error=False, fill_value=(0, 0)
        )
        interpol_e = interp1d(
            mz_e, norm(it_e), bounds_error=False, fill_value=(0, 0)
        )
        theory = interpol_t(mz_e) * 100
        exp = interpol_e(mz_e) * 100

        metrics["cosine"] = distance.cosine(theory, exp)
        # TODO: improve and add other statistics calculations
        return metrics
