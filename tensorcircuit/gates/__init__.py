import os
import sys
import types
from copy import deepcopy
from functools import lru_cache
from importlib import import_module, util
from typing import Any, Optional, List, Union, Dict, Tuple, TypedDict

import numpy as np

try:
    from numpy import ComplexWarning  # type: ignore
except ImportError:  # np2.0 compatibility
    from numpy.exceptions import ComplexWarning  # type: ignore

import tensornetwork as tn

from tensorcircuit.cons import backend, dtypestr, npdtype


__all__ = [
    "get_gate_module",
    "matrix_for_gate",
    "bmatrix",
    "Gate",
    "array_to_tensor",
    "num_to_tensor",
]


class GateDict(TypedDict):
    sgates: List[str]
    vgates: List[str]
    mpogates: List[str]
    gate_aliases: List[List[str]]


_qubit_gates: GateDict = {
    "sgates": ["i", "x", "y", "z", "h", "t", "s", "td", "sd", "wroot"]
    + ["cnot", "cz", "swap", "cy", "ox", "oy", "oz"]
    + ["toffoli", "fredkin"],
    "vgates": [
        "r",
        "cr",
        "u",
        "cu",
        "rx",
        "ry",
        "rz",
        "phase",
        "rxx",
        "ryy",
        "rzz",
        "cphase",
        "crx",
        "cry",
        "crz",
        "orx",
        "ory",
        "orz",
        "iswap",
        "any",
        "exp",
        "exp1",
    ],
    "mpogates": ["multicontrol", "mpo"],
    "gate_aliases": [
        ["cnot", "cx"],
        ["fredkin", "cswap"],
        ["toffoli", "ccnot"],
        ["toffoli", "ccx"],
        ["any", "unitary"],
        ["sd", "sdg"],
        ["td", "tdg"],
    ],
}

_qudit_gates: GateDict = {
    "sgates": ["i", "x", "z", "h", "s"],
    "vgates": ["u8", "cphase", "csum", "any"],
    "mpogates": [],  # ["mpo"],
    "gate_aliases": [
        ["any", "unitary"],
    ],
}

_FILE = os.path.join(os.path.dirname(__file__), "qudit_impl.py")

Tensor = Any
Array = Any
Operator = Any  # QuOperator


def merge_gate_dicts(
    d1: GateDict, d2: GateDict
) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    sgates = sorted(set(d1["sgates"]) | set(d2["sgates"]))
    vgates = sorted(set(d1["vgates"]) | set(d2["vgates"]))
    mpogates = sorted(set(d1["mpogates"]) | set(d2["mpogates"]))

    seen: set[tuple[str, ...]] = set()
    gate_aliases: List[List[str]] = []
    for pair in d1["gate_aliases"] + d2["gate_aliases"]:
        tup = tuple(pair)
        if tup not in seen:
            seen.add(tup)
            gate_aliases.append(list(tup))

    return sgates, vgates, mpogates, gate_aliases


sgates, vgates, mpogates, gate_aliases = merge_gate_dicts(_qubit_gates, _qudit_gates)


_EXPORTED_NAMES: set[str] = set()


def _populate_namespace(mod: types.ModuleType) -> None:
    """
    Export non-dunder attributes from the gate implementation module to the current module namespace,
    for example, allowing users to directly access `tc.gates._zz_matrix`.
    It will clean up old exports before switching dimensions to avoid residual pollution.
    """
    global _EXPORTED_NAMES
    for name in _EXPORTED_NAMES:
        globals().pop(name, None)
    _EXPORTED_NAMES = set()

    for name in dir(mod):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(mod, name)
        _EXPORTED_NAMES.add(name)


@lru_cache(maxsize=None)
def _load_qubit() -> types.ModuleType:
    return import_module(".qubit_impl", package=__name__)


@lru_cache(maxsize=None)
def _load_qudit(dim: int) -> types.ModuleType:
    mod_name = f"tensorcircuit.gates.qudit_impl_d{dim}"
    spec = util.spec_from_file_location(mod_name, _FILE)
    if spec is None:
        raise ImportError(f"Cannot build ModuleSpec for {mod_name} from {_FILE}")

    mod = util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"ModuleSpec has no loader for {mod_name} from {_FILE}")

    sys.modules[mod_name] = mod
    loader.exec_module(mod)
    mod.meta_gate(dim), mod.meta_vgate(dim)
    return mod


def set_gates_for(dim: Optional[int] = None, export: bool = True) -> types.ModuleType:
    """
    Select and load the specified dimension gates implementation module.
    When export=True (default),
     it will export the symbols to the tensorcircuit.gates namespace (all namespaces will be polluted);
    when export=False,
     it will only return the module object without modifying the global namespace (suitable for multiple d coexistence).
    """
    dim = 2 if dim is None else dim
    if not isinstance(dim, int) or dim < 2:
        raise ValueError("Dimension must be an integer >=2.")

    mod = _load_qubit() if dim == 2 else _load_qudit(dim)
    if export:
        _populate_namespace(mod)

    return mod


def get_gate_module(d: int) -> types.ModuleType:
    return set_gates_for(d, export=False)


class Gate(tn.Node):  # type: ignore
    """
    Wrapper of tn.Node, quantum gate
    """

    def __repr__(self) -> str:
        """Formatted output of Gate

        :Example:

        >>> tc.gates.ry(0.5)
        >>> # OR
        >>> print(repr(tc.gates.ry(0.5)))
        Gate(
            name: '__unnamed_node__',
            tensor:
                <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
                array([[ 0.9689124 +0.j, -0.24740396+0.j],
                    [ 0.24740396+0.j,  0.9689124 +0.j]], dtype=complex64)>,
            edges: [
                Edge(Dangling Edge)[0],
                Edge(Dangling Edge)[1]
            ])
        """
        sp = " " * 4
        edges = self.get_all_edges()
        edges_text = [edge.__repr__().replace("\n", "").strip() for edge in edges]
        edges_out = f"[" + f"\n{sp * 2}" + f",\n{sp * 2}".join(edges_text) + f"\n{sp}]"
        tensor_out = f"\n{sp * 2}" + self.tensor.__repr__().replace("\n", f"\n{sp * 2}")
        return (
            f"{self.__class__.__name__}(\n"
            f"{sp}name: {self.name!r},\n"
            f"{sp}tensor:{tensor_out},\n"
            f"{sp}edges: {edges_out})"
        )

    def copy(self, conjugate: bool = False) -> "Gate":
        result = super().copy(conjugate=conjugate)
        result.__class__ = Gate
        return result  # type: ignore


def num_to_tensor(*num: Union[float, Tensor], dtype: Optional[str] = None) -> Any:
    r"""
    Convert the inputs to Tensor with specified dtype.

    :Example:

    >>> from tensorcircuit.gates import num_to_tensor
    >>> # OR
    >>> from tensorcircuit.gates import array_to_tensor
    >>>
    >>> x, y, z = 0, 0.1, np.array([1])
    >>>
    >>> tc.set_backend('numpy')
    numpy_backend
    >>> num_to_tensor(x, y, z)
    [array(0.+0.j, dtype=complex64), array(0.1+0.j, dtype=complex64), array([1.+0.j], dtype=complex64)]
    >>>
    >>> tc.set_backend('tensorflow')
    tensorflow_backend
    >>> num_to_tensor(x, y, z)
    [<tf.Tensor: shape=(), dtype=complex64, numpy=0j>,
     <tf.Tensor: shape=(), dtype=complex64, numpy=(0.1+0j)>,
     <tf.Tensor: shape=(1,), dtype=complex64, numpy=array([1.+0.j], dtype=complex64)>]
    >>>
    >>> tc.set_backend('pytorch')
    pytorch_backend
    >>> num_to_tensor(x, y, z)
    [tensor(0.+0.j), tensor(0.1000+0.j), tensor([1.+0.j])]
    >>>
    >>> tc.set_backend('jax')
    jax_backend
    >>> num_to_tensor(x, y, z)
    [DeviceArray(0.+0.j, dtype=complex64),
     DeviceArray(0.1+0.j, dtype=complex64),
     DeviceArray([1.+0.j], dtype=complex64)]

    :param num: inputs
    :type num: Union[float, Tensor]
    :param dtype: dtype of the output Tensors
    :type dtype: str, optional
    :return: List of Tensors
    :rtype: List[Tensor]
    """
    # TODO(@YHPeter): fix __doc__ for same function with different names

    l = []
    if dtype is None:
        dtype = dtypestr
    for n in num:
        if not backend.is_tensor(n):
            l.append(backend.cast(backend.convert_to_tensor(n), dtype=dtype))
        else:
            l.append(backend.cast(n, dtype=dtype))
    if len(l) == 1:
        return l[0]
    return l


array_to_tensor = num_to_tensor


def gate_wrapper(m: Tensor, n: Optional[str] = None) -> Gate:
    if n is None:
        n = "unknowngate"
    m = m.astype(npdtype)
    return Gate(deepcopy(m), name=n)


def matrix_for_gate(gate: Gate, tol: float = 1e-6) -> Tensor:
    r"""
    Convert Gate to numpy array.

    :Example:

    >>> gate = tc.gates.r_gate()
    >>> tc.gates.matrix_for_gate(gate)
        array([[1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]], dtype=complex64)

    :param gate: input Gate
    :type gate: Gate
    :return: Corresponding Tensor
    :rtype: Tensor
    """

    t = gate.tensor
    t = backend.reshapem(t)
    t = backend.numpy(t)
    t.real[abs(t.real) < tol] = 0.0
    t.imag[abs(t.imag) < tol] = 0.0
    return t


def bmatrix(a: Array) -> str:
    r"""
    Returns a :math:`\LaTeX` bmatrix.

    :Example:

    >>> gate = tc.gates.r_gate()
    >>> array = tc.gates.matrix_for_gate(gate)
    >>> array
    array([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]], dtype=complex64)
    >>> print(tc.gates.bmatrix(array))
    \begin{bmatrix}    1.+0.j & 0.+0.j\\    0.+0.j & 1.+0.j \end{bmatrix}

    Formatted Display:

    .. math::
        \begin{bmatrix}    1.+0.j & 0.+0.j\\    0.+0.j & 1.+0.j \end{bmatrix}

    :param a: 2D numpy array
    :type a: np.array
    :raises ValueError: ValueError("bmatrix can at most display two dimensions")
    :return: :math:`\LaTeX`-formatted string for bmatrix of the array a
    :rtype: str
    """
    #   Adopted from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix/17131750

    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["    " + " & ".join(l.split()) + r"\\" for l in lines]
    rv[-1] = rv[-1][:-2]
    rv += [r" \end{bmatrix}"]
    return "".join(rv)


# --- Default behavior: Register and export qubit gates when importing tensorcircuit.gates ---
try:
    _DEFAULT_GATE_MODULE: types.ModuleType = set_gates_for(2, export=True)
except Exception:
    pass
