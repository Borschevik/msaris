from re import sub

import matplotlib as mpl
import numpy as np


replace_dict = {
    r"_(\d+)": "",
    r"([^0-9])1([^0-9])": r"\g<1>\g<2>",
    r"([^0-9])1$": r"\g<1>",
    r"(\d+)": r"_{\g<1>}",
    r"([-+])": r"^\g<1>",
}


def convert_into_formula_for_plot(value: str) -> str:
    """
    Converting formula to format to plot formula in correct notation

    :param value: formula value
    :return:
    """
    for pattern, replace in replace_dict.items():
        value = sub(pattern, replace, value)
    return f"${value}$"


def color_fader(c1: str, c2: str, *, mix: float = 0.0) -> list:
    """
    Calculate fading gradient
    from one colour to another in RGB byte format

    :param c1: the first color
    :param c2: the second calor
    :param mix: int number of spectra to get colors

    :return: list from colors
    """
    first1 = np.array(mpl.colors.to_rgb(c1))
    second2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * first1 + mix * second2)
