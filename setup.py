"""Setup script."""
from setuptools import setup

setup(
    name="bandu_stacking",
    version="0.1.0",
    packages=[],
    install_requires=["pybullet", "gym", "numpy"],
    include_package_data=True,
    extras_require={
        "develop": [
            "black",
            "docformatter",
            "isort",
            "mypy",
            "pylint>=2.14.5",
            "pytest-pylint>=0.18.0",
        ]
    },
)
