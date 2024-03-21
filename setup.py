"""Setup script."""

from setuptools import setup

setup(
    name="bandu_stacking",
    version="0.1.0",
    packages=[],
    include_package_data=True,
    install_requires=[
        "pylint>=2.14.5",
        "pytest-pylint>=0.18.0",
        "scipy==1.12.0",
        "pyzmq==25.1.2",
        "trimesh==4.2.0",
        "tqdm==4.66.2",
        "imageio==2.34.0",
        "docformatter==1.7.5",
        "black==24.3.0",
        "isort==5.13.2",
        "trimesh==4.2.0",
        "ghalton==0.6.2",
        "sbi==0.22.0",
    ],
)
