from __future__ import annotations

from functools import lru_cache, reduce, partial
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import tensornetwork as tn

from .utils import arg_alias
from .basecircuit import BaseCircuit
from .circuit import Circuit
from .quantum import QuOperator
from .quditGates import (
    _i_matrix_func,
    _x_matrix_func,
    _z_matrix_func,
    _h_matrix_func,
    _u8_matrix_func,
    _cphase_matrix_func,
    _csum_matrix_func,
)


Tensor = Any


class QuditCircuit:
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

    @lru_cache(maxsize=None)
    def _cached_single(
        self, gate: str, d: int, omega: float, extra: Tuple[Any, ...] = ()
    ) -> Tensor:
        if gate == "I":
            return _i_matrix_func(d)
        if gate == "X":
            return _x_matrix_func(d)
        if gate == "Z":
            return _z_matrix_func(d, omega)
        if gate == "H":
            return _h_matrix_func(d, omega)
        if gate == "U8":
            gamma, z, eps = extra
            return _u8_matrix_func(d, gamma, z, eps)
        raise ValueError(f"Unknown single-qudit gate: {gate}")

    def _apply_single(
        self,
        index: int,
        gate: str,
        name: Optional[str] = None,
        extra: Tuple[Any, ...] = (),
    ) -> None:
        mat = self._cached_single(gate, self._d, self._omega, extra)
        self._circ.unitary(index, unitary=mat, name=name or gate, dim=self._d)  # type: ignore

    def _tensor(self, gates: Sequence[Tuple[str, Tuple[Any, ...]]]) -> Tensor:
        mats = [
            self._cached_single(g, self._d, self._omega, extra) for (g, extra) in gates
        ]
        return reduce(np.kron, mats)

    def _apply_multi(
        self,
        *indices: Sequence[int],
        gates: Sequence[Tuple[str, Tuple[Any, ...]]],
        name: str,
    ) -> None:
        if len(indices) != len(gates):
            raise ValueError("len(indices) must equal len(gates)")
        mat = self._tensor(gates)
        self._circ.unitary(*indices, unitary=mat, name=name, dim=self._d)  # type: ignore

    def i(self, index: int) -> None:
        self._apply_single(index, "I")

    def x(self, index: int) -> None:
        self._apply_single(index, "X")

    def z(self, index: int) -> None:
        self._apply_single(index, "Z")

    def h(self, index: int) -> None:
        self._apply_single(index, "H")

    def u8(
        self, index: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
    ) -> None:
        self._apply_single(index, "U8", extra=(gamma, z, eps), name="U8")

    def ii(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("I", ()), ("I", ())], name="II")

    def xx(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("X", ()), ("X", ())], name="XX")

    def zz(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("Z", ()), ("Z", ())], name="ZZ")

    def ix(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("I", ()), ("X", ())], name="IX")

    def iz(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("I", ()), ("Z", ())], name="IZ")

    def xi(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("X", ()), ("I", ())], name="XI")

    def zi(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("Z", ()), ("I", ())], name="ZI")

    def xz(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("X", ()), ("Z", ())], name="XZ")

    def zx(self, *indices: int) -> None:
        self._apply_multi(*indices, gates=[("Z", ()), ("X", ())], name="ZX")

    def cphase(self, *indices: int, cv: Optional[int] = None) -> None:
        mat = _cphase_matrix_func(self._d, cv=cv)
        self._circ.unitary(*indices, unitary=mat, name="CPHASE", dim=self._d)  # type: ignore

    def csum(self, *indices: int, cv: Optional[int] = None) -> None:
        mat = _csum_matrix_func(self._d, cv=cv)
        self._circ.unitary(*indices, unitary=mat, name="CSUM", dim=self._d)  # type: ignore

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


def expectation(
    *ops: Tuple[tn.Node, List[int]],
    ket: Tensor,
    d: Optional[int] = None,
    bra: Optional[Tensor] = None,
    conj: bool = True,
    normalization: bool = False,
) -> Tensor:
    from .circuit import expectation

    return expectation(
        *ops,
        ket=ket,
        d=d,
        bra=bra,
        conj=conj,
        normalization=normalization,
    )
