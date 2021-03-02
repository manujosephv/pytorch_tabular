#!/usr/bin/env python

"""The setup script."""
import os
from setuptools import setup, find_packages


def read_requirements(thelibFolder, filename):
    requirementPath = os.path.join(thelibFolder, filename)
    requirements = []
    if os.path.isfile(requirementPath):
        with open(requirementPath) as f:
            requirements = f.read().splitlines()
    return requirements


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("docs/history.md") as history_file:
    history = history_file.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))

requirements = read_requirements(thelibFolder, "requirements.txt")
requirements_testing = read_requirements(thelibFolder, "requirements_testing.txt")
requirements_extra = read_requirements(thelibFolder, "requirements_extra.txt")

# setup_requirements = ['pytest-runner', ]

test_requirements = requirements_testing

setup(
    author="Manu Joseph",
    author_email="manujosephv@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A standard framework for using Deep Learning for tabular data",
    install_requires=requirements,
    extras_require={"all": requirements_extra},
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="pytorch, tabular, pytorch-lightning, neural network",
    name="pytorch_tabular",
    packages=find_packages(include=["pytorch_tabular", "pytorch_tabular.*"]),
    # setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/manujosephv/pytorch_tabular",
    version="0.3.0",
    zip_safe=False,
)
