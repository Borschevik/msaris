"""
    Clusterization algorithms to find isotope patterns ans estimate their averaged mass
"""
from copy import deepcopy
from typing import (
    Any,
    Literal,
    Union,
)

import numpy as np
from scipy.interpolate import interp1d

from msaris.reader.preprocessing import filter_intensities
from msaris.utils.cluster_utils import find_indexes
from msaris.utils.distributions_util import generate_gauss_distribution
from msaris.utils.intensities_util import get_spectrum_by_close_values


class ClusterSearch:
    """
    Inspired by Senko 1995 and Patterson routine algorithm so searching clusters
    And after each step removing them from initial spectrum until all clusters not found
    Iterative search performed to find all significant clusters from selecting peaks
    and range for selected maximum peak
    """

    def __init__(
        self,
        resolution: int = 10 ** 5,
        min_peaks: int = 1,
        cluster_width: int = 8,  # TODO: add option to define automatically
        cluster_min_dist: int = 5,  # TODO: add support for CI
    ):
        self.resolution = resolution
        self.min_peaks = min_peaks
        self.cluster_width = cluster_width
        self.cluster_min_dist = cluster_min_dist

    def _find_delta_mz(
        self, mz: np.array, it: np.array, charge: int
    ) -> np.array:
        """
        Perform patterson routine for selected diaposon of delta of list

        :param mz: m/z of mass spectrometry
        :param it: intensities of provided spectre

        :return: indexes which correspond to potential
        cluster with defined charge (default = 1)
        """
        interpol = interp1d(mz, it)
        delta_m = np.linspace(0, 10, self.resolution)
        results = []
        selected = delta_m[np.where(np.round(1 / delta_m) == charge)]
        for i in selected:
            mz_f, _, _, _ = get_spectrum_by_close_values(
                mz, it, left_border=mz[0], right_border=mz[-1], eps=1
            )
            plus = interpol(np.add(mz_f, i))
            minus = interpol(np.add(mz_f, -i))
            results.append(np.sum(plus * minus))
        return selected[results.index(max(results))]

    def _merge_close_masses(self, clusters: dict) -> dict:
        """
        Merge close index
        In future would be omitted

        :param clusters: merge close factors
        :return: merged dicts
        """

        sorted_mass = np.array(sorted(clusters))
        clusters = deepcopy(clusters)
        dist: int = np.sqrt(2 * (pow(self.cluster_min_dist, 2)))
        while (np.diff(sorted_mass) <= self.cluster_min_dist).any():
            visited = []
            for mass in sorted_mass:
                index = find_indexes(sorted_mass, mass, dist)
                if len(index) > 1:
                    close_masses = sorted_mass[index]
                    new_mz, new_it = (
                        np.array([]),
                        np.array([]),
                    )
                    for close_mass in close_masses:
                        if close_mass in visited:
                            continue
                        cluster = clusters[close_mass]
                        new_mz = np.concatenate(
                            (new_mz, cluster[0]), axis=None
                        )
                        new_it = np.concatenate(
                            (new_it, cluster[1]), axis=None
                        )
                        clusters.pop(close_mass)
                        visited.append(close_mass)
                    ind_sorted = np.argsort(new_mz)
                    new_mass = np.mean(close_masses)
                    if ind_sorted.any():
                        clusters[new_mass] = (
                            new_mz[ind_sorted],
                            new_it[ind_sorted],
                        )
            sorted_mass = np.array(sorted(clusters))

        return {round(k, 3): clusters[k] for k in sorted(clusters)}

    def find(
        self,
        mz: np.array,
        it: np.array,
        *,
        charge=1,
        threshold: float = 0.0,
        tolerance: float = 0.2
    ) -> dict:
        """
        Perform running over m/z and intensity values to find clusters

        :param mz: m/z of mass spectrometry
        :param it: intensities of provided spectre
        :param charge: isotope pattern charge
        :param threshold: threshold to filter intensities
        :param tolerance: tolerance for calculating delta of charge

        :return: dict of searched clusters
        """
        find_clusters = {}
        mz, it = filter_intensities(mz, it, threshold)
        while mz.shape[0] > 1:
            max_peak_search = mz[np.argmax(it)]

            mz_x, it_y, left, _ = get_spectrum_by_close_values(
                mz,
                it,
                max_peak_search,
                max_peak_search,
                eps=self.cluster_width,
            )
            mz_x, it_y, mass = generate_gauss_distribution(mz_x, it_y)
            delta_mz = self._find_delta_mz(mz_x, it_y, charge)
            difference = np.abs(mz_x - max_peak_search) / delta_mz
            index_delta = np.where(
                (np.abs(np.round(difference) - difference)) < tolerance
            )[0]

            if len(index_delta) > self.min_peaks:
                mz_x, it_y, mass = generate_gauss_distribution(
                    mz[left + index_delta],
                    it[left + index_delta],
                )
                find_clusters[np.round(mass, 3)] = (mz_x, it_y)

            mz = np.delete(mz, left + index_delta)
            it = np.delete(it, left + index_delta)

        if self.cluster_min_dist is not None:
            return self._merge_close_masses(find_clusters)
        return find_clusters


class MaxClustering:
    """
    Simple realization for clusters based on finding all possible max peak in clusters
    with defined window and providing them
    """

    def __init__(self, window: int = 8, threshold: float = 0.0):
        self.window = window
        self.threshold = threshold

    def find(
        self,
        mz: np.array,
        it: np.array,
        *,
        cluster_mod: Literal["max", "averaged"] = "averaged"
    ) -> dict:
        """
        Get list of cluster and parameters of max
        intensity peaks for window with selected resolution.
        :param mz: list of m/z defined for selected spectra
        :param it: list on intensities defined

        :returns: list of indexes for defined m/z
        """
        mass: Union[Any, float, None]
        find_clusters = {}
        if cluster_mod not in ["max", "averaged"]:
            raise ValueError("Cluster mod not max or averaged")
        mz, it = filter_intensities(mz, it, self.threshold)
        while mz.shape[0] > 1:
            max_peak_search = mz[np.argmax(it)]

            mz_x, it_y, left, right = get_spectrum_by_close_values(
                mz, it, max_peak_search, max_peak_search, eps=self.window
            )
            indexes = list(range(left, right))

            mz_f, it_f, mass = generate_gauss_distribution(mz_x, it_y)
            if cluster_mod == "max":
                mass = mz_f[np.argmax(it_f)]
            find_clusters[np.round(mass, 3)] = (
                mz_f,
                it_f,
            )
            mz = np.delete(mz, [indexes])
            it = np.delete(it, [indexes])

        return find_clusters
