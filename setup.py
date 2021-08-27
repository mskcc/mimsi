import os
import sys
from setuptools import setup, find_packages
from subprocess import check_output


def req_file(filename):
    """
    We're using a requirements.txt file so that pyup.io can use this for security checks
    """
    with open(filename) as f:
        content = f.readlines()
        content = filter(lambda x: not x.startswith("#"), content)
    return [x.strip() for x in content]


def most_recent_tag():
    """
    Get the most recent tag for the repo.
    """
    return (
        check_output(["git", "describe", "--tags"])
        .decode("utf-8")
        .strip()
        .split("-")
        .pop(0)
    )


setup(
    name="MiMSI",
    version="v0.4.3",
    description="A deep, multiple instance learning based classifier for identifying Microsatellite Instability from NGS",
    url="https://github.com/mskcc/mimsi",
    author="John Ziegler",
    author_email="zieglerj@mskcc.org",
    license="GNU General Public License v3.0",
    install_requires=req_file("requirements.txt"),
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 2.7",
    ],
    packages=find_packages(exclude=["tests*"]),
    py_modules=["analyze"],
    python_requires=">=2.7",
    package_data={"model": ["*.model"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "analyze = analyze:main",
            "create_data = data.generate_vectors.create_data:main",
            "evaluate_sample = main.evaluate_sample:main",
            "mi_msi_train_test = main.mi_msi_train_test:main",
        ]
    },
)
