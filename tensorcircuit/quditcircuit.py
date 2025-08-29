from functools import lru_cache, reduce, partial
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import tensornetwork as tn

from .utils import arg_alias
from .basecircuit import BaseCircuit
from .circuit import Circuit
from .quantum import QuOperator, QuVector
from .quditgates import (
    _i_matrix_func,
    _x_matrix_func,
    _y_matrix_func,
    _z_matrix_func,
    _h_matrix_func,
    _u8_matrix_func,
    _cphase_matrix_func,
    _csum_matrix_func,
    _rx_matrix_func,
    _ry_matrix_func,
    _rz_matrix_func,
    _rxx_matrix_func,
    _rzz_matrix_func,
)


Tensor = Any


SINGLE_BUILDERS = {
    "I": (("none",), lambda d, omega, **kw: _i_matrix_func(d)),
    "X": (("none",), lambda d, omega, **kw: _x_matrix_func(d)),
    "Y": (("none",), lambda d, omega, **kw: _y_matrix_func(d, omega)),
    "Z": (("none",), lambda d, omega, **kw: _z_matrix_func(d, omega)),
    "H": (("none",), lambda d, omega, **kw: _h_matrix_func(d, omega)),
    "RX": (
        ("theta", "j", "k"),
        lambda d, omega, **kw: _rx_matrix_func(d, kw["theta"], kw["j"], kw["k"]),
    ),
    "RY": (
        ("theta", "j", "k"),
        lambda d, omega, **kw: _ry_matrix_func(d, kw["theta"], kw["j"], kw["k"]),
    ),
    "RZ": (
        ("theta", "j"),
        lambda d, omega, **kw: _rz_matrix_func(d, kw["theta"], kw["j"]),
    ),
    "U8": (
        ("gamma", "z", "eps"),
        lambda d, omega, **kw: _u8_matrix_func(
            d, kw["gamma"], kw["z"], kw["eps"], omega
        ),
    ),
}

TWO_BUILDERS = {
    "RXX": (
        ("theta", "j1", "k1", "j2", "k2"),
        lambda d, omega, **kw: _rxx_matrix_func(
            d, kw["theta"], kw["j1"], kw["k1"], kw["j2"], kw["k2"]
        ),
    ),
    "RZZ": (("theta",), lambda d, omega, **kw: _rzz_matrix_func(d, kw["theta"])),
    "CPHASE": (("cv",), lambda d, omega, **kw: _cphase_matrix_func(d, kw["cv"], omega)),
    "CSUM": (("cv",), lambda d, omega, **kw: _csum_matrix_func(d, kw["cv"])),
}


@lru_cache(maxsize=None)
def _cached_matrix(
    kind: str,
    name: str,
    d: int,
    omega: Optional[float] = None,
    key: Optional[tuple] = (),
) -> Tensor:
    """
    kind: "single" or "two"; key is a parameter tuple sorted by signature for caching
    """
    builders = SINGLE_BUILDERS if kind == "single" else TWO_BUILDERS
    _, builder = builders[name]
    sig = builders[name][0]
    kwargs = {k: v for k, v in zip(sig, key)}
    return builder(d, omega, **kwargs)


class QuditCircuit:
    r"""
    ``QuditCircuit`` class.

    Qudit quick example (d=3):
    .. code-block:: python

        c = tc.Circuit(2, d=3)
        c.h(0)
        c.x(1)
        c.csum(0, 1)
        c.sample(1024, format="count_dict_bin")
        # For d <= 36, string samples use base-d characters 0–9A–Z (A=10, ...).
    """

    is_dm = False

    def __init__(
        self,
        nqubits: int,
        dim: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
        split: Optional[Dict[str, Any]] = None,
    ):
        self._set_dim(dim=dim)
        self._nqubits = nqubits

        self._circ = Circuit(
            nqubits=nqubits,
            dim=dim,
            inputs=inputs,
            mps_inputs=mps_inputs,
            split=split,
            qudit=True,
        )
        self._omega = np.exp(2j * np.pi / self._d)
        self.circuit_param = self._circ.circuit_param

    def _set_dim(self, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 2:
            raise ValueError(
                f"QuditCircuit is only for qudits (dim>=3). "
                f"You passed dim={dim}. For qubits, please use `Circuit` instead."
            )
        # Require integer d>=2; current string-encoded IO supports d<=36 (0–9A–Z digits).
        if dim > 36:
            raise NotImplementedError(
                "The Qudit interface is only supported for dimension < 36 now."
            )
        self._d = dim

    @property
    def dim(self) -> int:
        return self._d

    @property
    def nqubits(self) -> int:
        return self._nqubits

    def _apply_gate(self, *indices, name: str, **kwargs):
        if len(indices) == 1 and name in SINGLE_BUILDERS:
            sig, _ = SINGLE_BUILDERS[name]
            key = tuple(kwargs.get(k) for k in sig if k != "none")
            mat = _cached_matrix(
                kind="single", name=name, d=self._d, omega=self._omega, key=key
            )
            self._circ.unitary(*indices, unitary=mat, name=name, dim=self._d)  # type: ignore
        elif len(indices) == 2 and name in TWO_BUILDERS:
            sig, _ = TWO_BUILDERS[name]
            key = tuple(kwargs.get(k) for k in sig if k != "none")
            mat = _cached_matrix(
                kind="two", name=name, d=self._d, omega=self._omega, key=key
            )
            self._circ.unitary(  # type: ignore
                *indices, unitary=mat, name=name, dim=self._d
            )
        else:
            raise ValueError(f"Unsupported gate/arity: {name} on {len(indices)} qudits")

    def any(self, *indices: int, unitary: Tensor, name: str = "any") -> None:
        self._circ.unitary(*indices, unitary=unitary, name=name, dim=self._d)

    unitary = any

    def i(self, index: int) -> None:
        self._apply_gate(index, name="I")

    def x(self, index: int) -> None:
        self._apply_gate(index, name="X")

    def y(self, index: int) -> None:
        self._apply_gate(index, name="Y")

    def z(self, index: int) -> None:
        self._apply_gate(index, name="Z")

    def h(self, index: int) -> None:
        self._apply_gate(index, name="H")

    def u8(
        self, index: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
    ) -> None:
        self._apply_gate(index, name="U8", extra=(gamma, z, eps))

    def rx(self, index: int, theta: float, j: int = 0, k: int = 1):
        self._apply_gate(index, name="RX", theta=theta, j=j, k=k)

    def ry(self, index: int, theta: float, j: int = 0, k: int = 1):
        self._apply_gate(index, name="RY", theta=theta, j=j, k=k)

    def rz(self, index: int, theta: float, j: int = 0):
        self._apply_gate(index, name="RZ", theta=theta, j=j)

    def rxx(
        self,
        *indices: int,
        theta: float,
        j1: int = 0,
        k1: int = 1,
        j2: int = 0,
        k2: int = 1,
    ):
        self._apply_gate(*indices, name="RXX", theta=theta, j1=j1, k1=k1, j2=j2, k2=k2)

    def rzz(self, *indices: int, theta: float):
        self._apply_gate(*indices, name="RZZ", theta=theta)

    def cphase(self, *indices: int, cv: Optional[int] = None) -> None:
        self._apply_gate(*indices, name="CPHASE", cv=cv)

    def csum(self, *indices: int, cv: Optional[int] = None) -> None:
        self._apply_gate(*indices, name="CSUM", cv=cv)

    # Functional
    def wavefunction(self, form: str = "default") -> tn.Node.tensor:
        return self._circ.wavefunction(form)

    state = wavefunction

    def get_quoperator(self) -> QuOperator:
        return self._circ.quoperator()

    quoperator = get_quoperator

    get_circuit_as_quoperator = get_quoperator
    get_state_as_quvector = BaseCircuit.quvector

    def matrix(self) -> Tensor:
        return self._circ.matrix()

    def measure_reference(
        self, *index: int, with_prob: bool = False
    ) -> Tuple[str, float]:
        return self._circ.measure_reference(*index, with_prob=with_prob)

    def expectation(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        enable_lightcone: bool = False,
        nmc: int = 1000,
        status: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        return self._circ.expectation(
            *ops,
            reuse=reuse,
            enable_lightcone=enable_lightcone,
            noise_conf=None,
            nmc=nmc,
            status=status,
            **kws,
        )

    def measure_jit(
        self, *index: int, with_prob: bool = False, status: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self._circ.measure_jit(*index, with_prob=with_prob, status=status)

    measure = measure_jit

    def amplitude(self, l: Union[str, Tensor]) -> Tensor:
        return self._circ.amplitude(l)

    def probability(self) -> Tensor:
        return self._circ.probability()

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def sample(
        self,
        batch: Optional[int] = None,
        allow_state: bool = False,
        readout_error: Optional[Sequence[Any]] = None,
        format: Optional[str] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Tensor] = None,
        jittable: bool = True,
    ) -> Any:
        """
        A bug was reported in the JAX backend: by default integers use int32 precision.
         As a result, values like 3^29 (and even 3^19) exceed the representable range,
          causing errors during the conversion step in sample/count.
        """
        if format in ["sample_int", "count_tuple", "count_dict_int"]:
            raise NotImplementedError(
                "`int` representation is not friendly for d-dimensional systems."
            )
        return self._circ.sample(
            batch=batch,
            allow_state=allow_state,
            readout_error=readout_error,
            format=format,
            random_generator=random_generator,
            status=status,
            jittable=jittable,
        )

    def projected_subsystem(self, traceout: Tensor, left: Tuple[int, ...]) -> Tensor:
        return self._circ.projected_subsystem(
            traceout=traceout,
            left=left,
        )

    def replace_inputs(self, inputs: Tensor) -> None:
        return self._circ.replace_inputs(inputs)

    def mid_measurement(self, index: int, keep: int = 0) -> Tensor:
        return self._circ.mid_measurement(index, keep=keep)

    mid_measure = mid_measurement
    post_select = mid_measurement
    post_selection = mid_measurement

    def get_quvector(self) -> QuVector:
        return self._circ.quvector()

    quvector = get_quvector

    def replace_mps_inputs(self, mps_inputs: QuOperator) -> None:
        return self._circ.replace_mps_inputs(mps_inputs)
