from setuptools import find_packages, setup

exec(open("diffsptk/version.py").read())

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="diffsptk",
    version=__version__,
    description="Python module for speech signal processing",
    author="SPTK Working Group",
    author_email="takenori@sp.nitech.ac.jp",
    url="https://github.com/sp-nitech/diffsptk",
    packages=find_packages(exclude=("docs", "tests", "tools")),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="dsp speech processing sptk pytorch",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">= 3.8",
    install_requires=[
        "torch >= 1.10.0",
        "torchcrepe >= 0.0.16",
        "numpy",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "numpydoc",
            "pydata-sphinx-theme",
            "pytest",
            "pytest-cov",
            "sphinx",
            "twine",
        ],
    },
)
