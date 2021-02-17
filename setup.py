from setuptools import (
    find_packages,
    setup,
)

import msaris


NAME = "msaris"
AUTHOR = "Anton Bondarenko, Yulia Vlasova, Mikhail Polynski"
AUTHOR_EMAIL = (
    "emptybox1267@gmail.com, zenziver61@gmail.com, polynskimikhail@gmail.com"
)
KEYWORDS = ", ".join(
    [
        "mass spectrometry",
        "mass spec",
        "isotopic",
        "linear programming",
        "isotope pattern",
        "clusterization",
    ]
)
PACKAGES = find_packages()
DESCRIPTION = """
Package of classes, methods and function to
perform search of isotope patterns, their identification and
identifying their formulas
"""

setup(
    name=NAME,
    version=msaris.__version__,
    description=(
        "Python package to identify isotope patterns and their formulas"
    ),
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    url="https://github.com/Borschevik/msaris",
    packages=PACKAGES,
    license="MIT",
    python_requires=">3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    install_requires=[
        "pyopenms>=2.6.0",
        "molmass>=2020.6.10",
        "pytest>=6.2.2",
        "scipy>=1.6.0",
        "PuLP>=2.4",
        "multiprocess>=0.70.11.1",
        "openms>=3.6.1",
        "pandas>=1.2.0",
        "numpy>=1.19.5",
        "typer>=0.3.2",
        "IsoSpecPy>=2.1.1",
    ],
    keywords=KEYWORDS,
)
