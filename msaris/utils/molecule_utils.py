from re import sub


replace_dict = {
    r"_(\d+)": "",
    r"1([^0-9]+\s*)": r"\g<1>",
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
