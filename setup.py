import setuptools

from tensorcircuit import __version__, __author__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tensorcircuit-ng",
    version=__version__,
    author=__author__,
    author_email="znfesnpbh@gmail.com",
    description="High performance unified quantum computing framework for the NISQ era",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorcircuit/tensorcircuit-ng",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "tensornetwork-ng", "networkx"],
    extras_require={
        "tensorflow": ["tensorflow"],
        "jax": ["jax", "jaxlib"],
        "torch": ["torch"],
        "qiskit": ["qiskit"],
        "cloud": ["qiskit", "mthree<2.8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
