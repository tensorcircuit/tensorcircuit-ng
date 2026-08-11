"""Smoke test TensorCircuit after installing only its required dependencies."""

import importlib.util

import numpy as np

import tensorcircuit as tc

OPTIONAL_MODULES = ("tensorflow", "jax", "torch", "cupy", "qiskit", "cirq", "stim")


def main() -> None:
    installed_optional_modules = [
        name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None
    ]
    assert not installed_optional_modules, installed_optional_modules

    assert tc.__version__
    assert callable(tc.compiler.lightcone_compile)
    tc.about()

    circuit = tc.Circuit(1)
    circuit.h(0)
    np.testing.assert_allclose(
        np.asarray(circuit.wavefunction()),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
    )


if __name__ == "__main__":
    main()
