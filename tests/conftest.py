import sys
import os
import numpy as np
import pytest

# NOTE: If Intel OpenMP abort occurs on Windows (PyTorch + scipy DLL conflict),
# set KMP_DUPLICATE_LIB_OK=TRUE in the test runner environment, e.g.:
#   $env:KMP_DUPLICATE_LIB_OK = "TRUE"

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "ignore::tensorcircuit.utils.ExperimentalWarning"
    )


@pytest.fixture(scope="function")
def npb():
    tc.set_backend("numpy")
    yield
    tc.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def tfb():
    tc.set_backend("tensorflow")
    yield
    tc.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def jaxb():
    try:
        tc.set_backend("jax")
        yield
        tc.set_backend("numpy")

    except ImportError as e:
        print(e)
        tc.set_backend("numpy")
        pytest.skip("****** No jax backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def torchb():
    try:
        tc.set_backend("pytorch")
        yield
        tc.set_backend("numpy")
    except ImportError as e:
        print(e)
        tc.set_backend("numpy")
        pytest.skip("****** No torch backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def cpb():
    try:
        tc.set_backend("cupy")
        yield
        tc.set_backend("numpy")
    except ImportError as e:
        print(e)
        tc.set_backend("numpy")
        pytest.skip("****** No cupy backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def highp():
    tc.set_dtype("complex128")
    yield
    tc.set_dtype("complex64")


@pytest.fixture
def with_readout_errors():
    def apply(probs, qubits, errors):
        # the readout distribution of the classical bits, bit k reading qubits[k]
        p = probs.reshape([2] * len(qubits))
        for k, q in enumerate(qubits):
            e0, e1 = errors[q]
            m = np.array([[1 - e0, e1], [e0, 1 - e1]])
            p = np.moveaxis(np.tensordot(m, p, axes=[[1], [k]]), 0, k)
        return p.reshape(-1)

    return apply
