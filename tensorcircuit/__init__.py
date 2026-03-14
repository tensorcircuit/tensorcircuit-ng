__version__ = "1.5.0"
__author__ = "TensorCircuit-NG Authors"
__creator__ = "refraction-ray"

import importlib
from typing import Any, List
from .utils import gpu_memory_share

gpu_memory_share()

from .about import about, cite
from .cons import (
    backend,
    set_backend,
    set_dtype,
    set_contractor,
    get_backend,
    get_dtype,
    get_contractor,
    set_function_backend,
    set_function_dtype,
    set_function_contractor,
    runtime_backend,
    runtime_dtype,
    runtime_contractor,
)  # prerun of set hooks
from . import gates
from . import quditgates
from . import basecircuit
from .gates import Gate
from .quditcircuit import QuditCircuit
from .analogcircuit import AnalogCircuit
from .circuit import Circuit, expectation
from .u1circuit import U1Circuit
from .mpscircuit import MPSCircuit
from .densitymatrix import DMCircuit as DMCircuit_reference
from .densitymatrix import DMCircuit2

DMCircuit = DMCircuit2  # compatibility issue to still expose DMCircuit2
DensityMatrixCircuit = DMCircuit

try:
    from .stabilizercircuit import StabilizerCircuit

    CliffordCircuit = StabilizerCircuit
    StabCircuit = StabilizerCircuit
except ModuleNotFoundError:
    pass

from .gates import num_to_tensor, array_to_tensor
from .vis import qir2tex, render_pdf
from . import interfaces
from . import templates
from . import results
from . import quantum
from .quantum import QuOperator, QuVector, QuAdjointVector, QuScalar
from . import compiler
from . import cloud
from . import fgs
from .fgs import FGSSimulator
from . import pauliprop
from .pauliprop import pauli_propagation
from . import timeevol

FGSCircuit = FGSSimulator

# lazy imports for heavy frameworks
# name: (module_relative_path, is_module)
_lazy_imports = {
    "keras": (".keras", True),
    "KerasLayer": (".keras", False),
    "KerasHardwareLayer": (".keras", False),
    "torchnn": (".torchnn", True),
    "TorchLayer": (".torchnn", False),
    "TorchHardwareLayer": (".torchnn", False),
}


def __getattr__(name: str) -> Any:
    if name in _lazy_imports:
        path, is_module = _lazy_imports[name]
        module = importlib.import_module(path, __package__)
        if is_module:
            attr = module
        else:
            attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError("module %s has no attribute %s" % (__name__, name))


def __dir__() -> List[str]:
    return sorted(set(globals().keys()).union(_lazy_imports.keys()))


try:
    import qiskit

    qiskit.QuantumCircuit.cnot = qiskit.QuantumCircuit.cx
    qiskit.QuantumCircuit.toffoli = qiskit.QuantumCircuit.ccx
    qiskit.QuantumCircuit.fredkin = qiskit.QuantumCircuit.cswap

    # amazing qiskit 1.0 nonsense...
except ModuleNotFoundError:
    pass

# just for fun
from .asciiart import set_ascii
