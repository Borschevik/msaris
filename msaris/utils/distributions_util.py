from typing import Tuple, Optional, List, Union, TypeVar

import numpy as np

MZ = TypeVar('MZ', np.ndarray, Tuple[np.ndarray, Optional[float]])


def generate_gauss_distribution(
    mass_out: list, intens_out: list, ppm: int = 50, resolution: int = 20
) -> Tuple[MZ, List[float], float]:
    """
    Generates gauss distribution for selected m/z and intensities

    :params mass_out: list of m/z
    :params intens_out: list of intensities
    :params ppm: int of peak width
    :params resolution: int for peaks resoltions

    :returns: tuple with m/z list, intensities list, weighted mass as a float
    """
    mass = 0.0
    norm_inens_out = intens_out / np.sum(intens_out)

    for i in range(len(mass_out)):
        mass += mass_out[i] * norm_inens_out[i]
    try:
        res = ppm * (mass) / 10 ** 6
    except ValueError:
        raise ValueError("Invalid ppm input!")

    num_points = (
        int((np.max(mass_out) - np.min(mass_out)) / res) + 1
    ) * resolution

    x: MZ = np.linspace(min(mass_out) - 1, np.max(mass_out) + 1, num_points)
    expected_value = mass_out
    gauss_filtered = [0.0] * len(x)
    for i, expected in enumerate(expected_value):
        deltaM = (ppm / 1000000) * mass_out[i]
        FWHM = deltaM
        sigma = float(
            FWHM / (2 * np.sqrt(2 * np.log(2)))
        )
        gauss_filtered += (
            (1 / (sigma * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x - expected_value[i]) / sigma) ** 2)
            * intens_out[i]
        )

    return (
        x,
        gauss_filtered,
        mass,
    )
