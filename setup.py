from setuptools import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_file = f.read()

setup(
    name="vnoise",
    version="0.1.0a0",
    description="Vectorized, pure-Python Perlin noise library",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Antoine Beyeler",
    email="abeyeler@ab-ware.com",
    url="https://github.com/vpype/vnoise",
    license=license_file,
    packages=["vnoise"],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19",
        "setuptools",
    ],
)
