"""
Quantum circuit: state simulator for **qudits** (d-level systems).

This module provides a high-level `QuditCircuit` API that mirrors `tensorcircuit.circuit.Circuit`
but targets qudits with dimension `3 <= d <= 36`.
For string-encoded samples/counts, digits use `0-9A-Z` where `A=10, ..., Z=35`.

.. note::
   For qubits (`d=2`) please use :class:`tensorcircuit.circuit.Circuit`.

"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union, Literal

import numpy as np
import tensornetwork as tn

from .gates import Gate
from .utils import arg_alias
from .basecircuit import BaseCircuit
from .circuit import Circuit
from .quantum import QuOperator, QuVector
from .quditgates import SINGLE_BUILDERS, TWO_BUILDERS


Tensor = Any
SAMPLE_FORMAT = Literal["sample_bin", "count_dict_bin"]


class QuditCircuit:
    r"""
    The `QuditCircuit` class provides a d-dimensional state-vector simulator and a thin wrapper
    around :class:`tensorcircuit.circuit.Circuit`, exposing a qudit-friendly API and docstrings.

    **Quick example (d=3):**

    >>> c = tc.QuditCircuit(2, dim=3)
    >>> c.h(0)
    >>> c.x(1)
    >>> c.csum(0, 1)
    >>> c.sample(1024, format="count_dict_bin")

    .. note::
       For `3 <= d <= 36`, string samples and count keys use base-`d` characters `0-9A-Z`
       (`A=10, ..., Z=35`).

    :param nqudits: Number of qudits (wires) in the circuit.
    :type nqudits: int
    :param dim: Qudit local dimension `d`. Must satisfy `3 <= d <= 36`.
    :type dim: int
    :param inputs: Optional initial state as a wavefunction.
    :type inputs: Optional[Tensor]
    :param mps_inputs: Optional initial state in MPS/MPO-like form.
    :type mps_inputs: Optional[QuOperator]
    :param split: Internal contraction/splitting configuration passed through to
                  :class:`~tensorcircuit.circuit.Circuit`.
    :type split: Optional[Dict[str, Any]]

    :var is_dm: Whether the simulator is a density-matrix simulator (`False` here).
    :vartype is_dm: bool
    :var dim: Property for the local dimension `d`.
    :vartype dim: int
    :var nqudits: Property for the number of qudits.
    :vartype nqudits: int
    """

    is_dm = False

    def __init__(
        self,
        nqudits: int,
        dim: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
        split: Optional[Dict[str, Any]] = None,
    ):
        self._set_dim(dim=dim)
        self._nqudits = nqudits

        self._circ = Circuit(
            nqubits=nqudits,
            dim=dim,
            inputs=inputs,
            mps_inputs=mps_inputs,
            split=split,
        )
        self._omega = np.exp(2j * np.pi / self._d)
        self.circuit_param = self._circ.circuit_param

    def _set_dim(self, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 2:
            raise ValueError(
                f"QuditCircuit is only for qudits (dim>=3). "
                f"You passed dim={dim}. For qubits, please use `Circuit` instead."
            )
        # Require integer d>=2; current string-encoded IO supports d<=36 (0-9A-Z digits).
        if dim > 36:
            raise NotImplementedError(
                "The Qudit interface is only supported for dimension < 36 now."
            )
        self._d = dim

    @property
    def dim(self) -> int:
        """dimension of the qudit circuit"""
        return self._d

    @property
    def nqudits(self) -> int:
        """qudit number of the circuit"""
        return self._nqudits

    def _apply_gate(self, *indices: int, name: str, **kwargs: Any) -> None:
        """
        Apply a single- or two-qudit unitary by name.

        The gate matrix is looked up by name in either :data:`SINGLE_BUILDERS` (single-qudit)
        or :data:`TWO_BUILDERS` (two-qudit). The registered builder is called with `(d, omega, **kwargs)`
        to produce the unitary, which is then applied at the given indices.

        :param indices: Qudit indices the gate acts on. One index -> single-qudit gate; two indices -> two-qudit gate.
        :type indices: int
        :param name: Gate name registered in the corresponding builder set.
        :type name: str
        :param kwargs: Extra parameters forwarded to the builder (matched by keyword).
        :type kwargs: Any
        :raises ValueError: If the name is unknown or the arity does not match the number of indices.
        """
        if len(indices) == 1 and name in SINGLE_BUILDERS:
            sig, builder = SINGLE_BUILDERS[name]
            extras = tuple(kwargs.get(k) for k in sig if k != "none")
            builder_kwargs = {k: v for k, v in zip(sig, extras)}
            mat = builder(self._d, self._omega, **builder_kwargs)
            self._circ.unitary(*indices, unitary=mat, name=name, dim=self._d)  # type: ignore
        elif len(indices) == 2 and name in TWO_BUILDERS:
            sig, builder = TWO_BUILDERS[name]
            extras = tuple(kwargs.get(k) for k in sig if k != "none")
            builder_kwargs = {k: v for k, v in zip(sig, extras)}
            mat = builder(self._d, self._omega, **builder_kwargs)
            self._circ.unitary(  # type: ignore
                *indices, unitary=mat, name=name, dim=self._d
            )
        else:
            raise ValueError(f"Unsupported gate/arity: {name} on {len(indices)} qudits")

    def any(self, *indices: int, unitary: Tensor, name: Optional[str] = None) -> None:
        """
        Apply an arbitrary unitary on one or two qudits.

        :param indices: Target qudit indices.
        :type indices: int
        :param unitary: Unitary matrix acting on the specified qudit(s), with shape `(d, d)` or `(d^2, d^2)`.
        :type unitary: Tensor
        :param name: Optional label stored in the circuit history.
        :type name: str
        """
        name = "any" if name is None else name
        self._circ.unitary(*indices, unitary=unitary, name=name, dim=self._d)  # type: ignore

    unitary = any

    def i(self, index: int) -> None:
        """
        Apply the generalized identity gate `I` on the given qudit.

        :param index: Qudit index.
        :type index: int
        """
        self._apply_gate(index, name="I")

    I = i

    def x(self, index: int) -> None:
        """
        Apply the generalized shift gate `X` on the given qudit (adds `+1 mod d`).

        :param index: Qudit index.
        :type index: int
        """
        self._apply_gate(index, name="X")

    X = x

    # def y(self, index: int) -> None:
    #     """
    #     Apply the Y gate on the given qudit index.
    #
    #     :param index: Qudit index to apply the gate on.
    #     :type index: int
    #     """
    #     self._apply_gate(index, name="Y")

    def z(self, index: int) -> None:
        """
        Apply the generalized phase gate `Z` on the given qudit (multiplies by `omega^k`).

        :param index: Qudit index.
        :type index: int
        """
        self._apply_gate(index, name="Z")

    Z = z

    def h(self, index: int) -> None:
        """
        Apply the generalized Hadamard-like gate `H` (DFT on `d` levels) on the given qudit.

        :param index: Qudit index.
        :type index: int
        """
        self._apply_gate(index, name="H")

    H = h

    def u8(
        self, index: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
    ) -> None:
        """
        Apply the parameterized single-qudit gate `U8`.

        :param index: Qudit index.
        :type index: int
        :param gamma: Gate parameter `gamma`.
        :type gamma: float
        :param z: Gate parameter `z`.
        :type z: float
        :param eps: Gate parameter `eps`.
        :type eps: float
        """
        self._apply_gate(index, name="U8", extra=(gamma, z, eps))

    U8 = u8

    def rx(self, index: int, theta: float, j: int = 0, k: int = 1) -> None:
        """
        Single-qudit rotation `RX` on a selected two-level subspace `(j, k)`.

        :param index: Qudit index.
        :type index: int
        :param theta: Rotation angle.
        :type theta: float
        :param j: Source level of the rotation subspace (0-based).
        :type j: int
        :param k: Target level of the rotation subspace (0-based).
        :type k: int
        :raises ValueError: If `j == k` or indices are outside `[0, d-1]`.
        """
        self._apply_gate(index, name="RX", theta=theta, j=j, k=k)

    RX = rx

    def ry(self, index: int, theta: float, j: int = 0, k: int = 1) -> None:
        """
        Single-qudit rotation `RY` on a selected two-level subspace `(j, k)`.

        :param index: Qudit index.
        :type index: int
        :param theta: Rotation angle.
        :type theta: float
        :param j: Source level of the rotation subspace (0-based).
        :type j: int
        :param k: Target level of the rotation subspace (0-based).
        :type k: int
        :raises ValueError: If `j == k` or indices are outside `[0, d-1]`.
        """
        self._apply_gate(index, name="RY", theta=theta, j=j, k=k)

    RY = ry

    def rz(self, index: int, theta: float, j: int = 0) -> None:
        """
        Single-qudit phase rotation `RZ` applied on level `j`.

        :param index: Qudit index.
        :type index: int
        :param theta: Phase rotation angle around Z.
        :type theta: float
        :param j: Level where the phase is applied (0-based).
        :type j: int
        :raises ValueError: If `j` is outside `[0, d-1]`.
        """
        self._apply_gate(index, name="RZ", theta=theta, j=j)

    RZ = rz

    def rxx(
        self,
        *indices: int,
        theta: float,
        j1: int = 0,
        k1: int = 1,
        j2: int = 0,
        k2: int = 1,
    ) -> None:
        """
        Two-qudit interaction `RXX` acting on subspaces `(j1, k1)` and `(j2, k2)`.

        :param indices: Two qudit indices `(q1, q2)`.
        :type indices: int
        :param theta: Interaction angle.
        :type theta: float
        :param j1: Source level of the first qudit subspace.
        :type j1: int
        :param k1: Target level of the first qudit subspace.
        :type k1: int
        :param j2: Source level of the second qudit subspace.
        :type j2: int
        :param k2: Target level of the second qudit subspace.
        :type k2: int
        :raises ValueError: If levels are invalid or the arity is not two.
        """
        self._apply_gate(*indices, name="RXX", theta=theta, j1=j1, k1=k1, j2=j2, k2=k2)

    RXX = rxx

    def rzz(
        self,
        *indices: int,
        theta: float,
        j1: int = 0,
        k1: int = 1,
        j2: int = 0,
        k2: int = 1,
    ) -> None:
        """
        Two-qudit interaction `RZZ` acting on subspaces `(j1, k1)` and `(j2, k2)`.

        :param indices: Two qudit indices `(q1, q2)`.
        :type indices: int
        :param theta: Interaction angle.
        :type theta: float
        :param j1: Source level of the first qudit subspace.
        :type j1: int
        :param k1: Target level of the first qudit subspace.
        :type k1: int
        :param j2: Source level of the second qudit subspace.
        :type j2: int
        :param k2: Target level of the second qudit subspace.
        :type k2: int
        :raises ValueError: If levels are invalid or the arity is not two.
        """
        self._apply_gate(*indices, name="RZZ", theta=theta, j1=j1, k1=k1, j2=j2, k2=k2)

    RZZ = rzz

    def cphase(self, *indices: int, cv: Optional[int] = None) -> None:
        """
        Apply a controlled-phase gate `CPHASE`.

        :param indices: Two qudit indices `(control, target)`.
        :type indices: int
        :param cv: Optional control value. If `None`, defaults to `1`.
        :type cv: Optional[int]
        :raises ValueError: If arity is not two.
        """
        self._apply_gate(*indices, name="CPHASE", cv=cv)

    CPHASE = cphase

    def csum(self, *indices: int, cv: Optional[int] = None) -> None:
        """
        Apply a generalized controlled-sum gate `CSUM` (a.k.a. qudit CNOT).

        :param indices: Two qudit indices `(control, target)`.
        :type indices: int
        :param cv: Optional control value. If `None`, defaults to `1`.
        :type cv: Optional[int]
        :raises ValueError: If arity is not two.
        """
        self._apply_gate(*indices, name="CSUM", cv=cv)

    cnot, CSUM, CNOT = csum, csum, csum

    # Functional
    def wavefunction(self, form: str = "default") -> tn.Node.tensor:
        """
        Compute the output wavefunction from the circuit.

        :param form: The str indicating the form of the output wavefunction.
            "default": [-1], "ket": [-1, 1], "bra": [1, -1]
        :type form: str, optional
        :return: Tensor with the corresponding shape.
        :rtype: Tensor
        """
        return self._circ.wavefunction(form)

    state = wavefunction

    def get_quoperator(self) -> QuOperator:
        """
        Get the ``QuOperator`` MPO like representation of the circuit unitary without contraction.

        :return: ``QuOperator`` object for the circuit unitary (open indices for the input state)
        :rtype: QuOperator
        """
        return self._circ.quoperator()

    quoperator = get_quoperator

    get_circuit_as_quoperator = get_quoperator
    get_state_as_quvector = BaseCircuit.quvector

    def matrix(self) -> Tensor:
        """
        Get the unitary matrix for the circuit irrespective with the circuit input state.

        :return: The circuit unitary matrix
        :rtype: Tensor
        """
        return self._circ.matrix()

    # def measure_reference(
    #     self, *index: int, with_prob: bool = False
    # ) -> Tuple[str, float]:
    #     return self._circ.measure_reference(*index, with_prob=with_prob)

    def expectation(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        enable_lightcone: bool = False,
        nmc: int = 1000,
        **kws: Any,
    ) -> Tensor:
        """
        Compute expectation(s) of local operators.

        :param ops: Pairs of `(operator_node, [sites])` specifying where each operator acts.
        :type ops: Tuple[tn.Node, List[int]]
        :param reuse: If True, then the wavefunction tensor is cached for further expectation evaluation,
            defaults to be true.
        :type reuse: bool, optional
        :param enable_lightcone: whether enable light cone simplification, defaults to False
        :type enable_lightcone: bool, optional
        :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
        :type nmc: int, optional
        :raises ValueError: "Cannot measure two operators in one index"
        :return: Tensor with one element
        :rtype: Tensor
        """
        return self._circ.expectation(
            *ops,
            reuse=reuse,
            enable_lightcone=enable_lightcone,
            noise_conf=None,
            nmc=nmc,
            **kws,
        )

    def measure_jit(
        self, *index: int, with_prob: bool = False, status: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement on the given site indices (computational basis).
        This method is jittable!

        :param index: Measure on which site (wire) index.
        :type index: int
        :param with_prob: If true, theoretical probability is also returned.
        :type with_prob: bool, optional
        :param status: external randomness, with shape [index], defaults to None
        :type status: Optional[Tensor]
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[Tensor, Tensor]
        """
        return self._circ.measure_jit(*index, with_prob=with_prob, status=status)

    measure = measure_jit

    def amplitude(self, l: Union[str, Tensor]) -> Tensor:
        r"""
        Return the amplitude for a given base-`d` string `l`.

        For state simulators, this computes :math:`\langle l \vert \psi \rangle`.
        For density-matrix simulators, it would compute :math:`\operatorname{Tr}(\rho \vert l \rangle \langle l \vert)`
        (note the square in magnitude differs between the two formalisms).

        **Example**

        >>> c = tc.QuditCircuit(2, dim=3)
        >>> c.x(0)
        >>> c.x(1)
        >>> c.amplitude("20")
        array(1.+0.j, dtype=complex64)
        >>> c.csum(0, 1, cv=2)
        >>> c.amplitude("21")
        array(1.+0.j, dtype=complex64)

        :param l: Bitstring in base-`d` using `0-9A-Z`.
        :type l: Union[str, Tensor]
        :return: Complex amplitude.
        :rtype: Tensor
        """
        return self._circ.amplitude(l)

    def probability(self) -> Tensor:
        """
        Get the length-`d^n` probability vector over the computational basis.

        :return: Probability vector of shape `[dim^n]`.
        :rtype: Tensor
        """
        return self._circ.probability()

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def sample(
        self,
        batch: Optional[int] = None,
        allow_state: bool = False,
        readout_error: Optional[Sequence[Any]] = None,
        format: Optional[SAMPLE_FORMAT] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Tensor] = None,
        jittable: bool = True,
    ) -> Any:
        r"""
        Batched sampling from the circuit or final state.

        :param batch: Number of samples. If `None`, returns a single draw.
        :type batch: Optional[int]
        :param allow_state: If `True`, sample from the final state (when memory allows). Prefer `True` for speed.
        :type allow_state: bool
        :param readout_error: Optional readout error model.
        :type readout_error: Optional[Sequence[Any]]
        :param format: Output format. See :py:meth:`tensorcircuit.quantum.measurement_results`.
                       Supported formats for qudits include:

                "count_vector": # np.array([2, 0, 0, 0])

                "count_dict_bin": # {"00": 2, "01": 0, "10": 0, "11": 0}
                    for cases :math:`d\in [11, 36]`, use 0-9A-Z digits (e.g., 'A' -> 10, ..., 'Z' -> 35);

        :type format: Optional[str]
        :param random_generator: random generator,  defaults to None
        :type random_generator: Optional[Any], optional
        :param status: external randomness given by tensor uniformly from [0, 1],
            if set, can overwrite random_generator, shape [batch] for `allow_state=True`
            and shape [batch, nqudits] for `allow_state=False` using perfect sampling implementation
        :type status: Optional[Tensor]
        :param jittable: when converting to count, whether keep the full size. if false, may be conflict
            external jit, if true, may fail for large scale system with actual limited count results
        :type jittable: bool, defaults true
        :return: List (if batch) of tuple (binary configuration tensor and corresponding probability)
            if the format is None, and consistent with format when given
        :rtype: Any
        :raises NotImplementedError: For integer-based output formats not suitable for qudits.
        """
        if format in ["sample_int", "count_tuple", "count_dict_int", "count_vector"]:
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
        """
        remaining wavefunction or density matrix on sites in `left`, while fixing the other
        sites to given digits as indicated by `traceout`.

        :param traceout: A tensor encoding digits (0..d-1) for sites to be fixed; jittable.
        :type traceout: Tensor
        :param left: Tuple of site indices to keep (non-jittable argument).
        :type left: Tuple[int, ...]
        :return: Remaining wavefunction or density matrix on the kept sites.
        :rtype: Tensor
        """
        return self._circ.projected_subsystem(
            traceout=traceout,
            left=left,
        )

    def replace_inputs(self, inputs: Tensor) -> None:
        """
        Replace the input state with the circuit structure unchanged.

        :param inputs: Input wavefunction.
        :type inputs: Tensor
        """
        return self._circ.replace_inputs(inputs)

    def mid_measurement(self, index: int, keep: int = 0) -> Tensor:
        """
        Mid-circuit Z-basis post-selection.

        The returned state is **not normalized**; normalize manually if needed.

        :param index: Qudit index where post-selection is applied.
        :type index: int
        :param keep: Post-selected digit in `{0, ..., d-1}`.
        :type keep: int
        :return: Unnormalized post-selected state.
        :rtype: Tensor
        """
        return self._circ.mid_measurement(index, keep=keep)

    mid_measure = mid_measurement
    post_select = mid_measurement
    post_selection = mid_measurement

    def get_quvector(self) -> QuVector:
        """
        Get the representation of the output state in the form of ``QuVector``
        while maintaining the circuit uncomputed

        :return: ``QuVector`` representation of the output state from the circuit
        :rtype: QuVector
        """
        return self._circ.quvector()

    quvector = get_quvector

    def replace_mps_inputs(self, mps_inputs: QuOperator) -> None:
        """
        Replace the input state (keeps circuit structure) using an MPS/MPO-like representation.

        **Example**

        >>> c = tc.QuditCircuit(2, dim=3)
        >>> c.x(0)
        >>> c2 = tc.QuditCircuit(2, dim=3, mps_inputs=c.quvector())
        >>> c2.x(0); c2.wavefunction()
        array([...], dtype=complex64)
        >>> c3 = tc.QuditCircuit(2, dim=3)
        >>> c3.x(0)
        >>> c3.replace_mps_inputs(c.quvector()); c3.wavefunction()
        array([...], dtype=complex64)

        :param mps_inputs: (Nodes, dangling Edges) for a MPS like initial wavefunction.
        :type mps_inputs: Tuple[Sequence[Gate], Sequence[Edge]]
        """
        return self._circ.replace_mps_inputs(mps_inputs)

    def expectation_before(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        **kws: Any,
    ) -> List[tn.Node]:
        """
        Build (but do not contract) the tensor network for expectation evaluation.

        :param reuse: _description_, defaults to True
        :type reuse: bool, optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: List[tn.Node]
        """
        return self._circ.expectation_before(*ops, reuse=reuse, **kws)

    def amplitude_before(self, l: Union[str, Tensor]) -> List[Gate]:
        r"""
        Return the (uncontracted) tensor network nodes for the amplitude of configuration `l`.

        For state simulators, this corresponds to :math:`\langle l \vert \psi \rangle`.
        For density-matrix simulators,
        it would correspond to :math:`\operatorname{Tr}(\rho \vert l \rangle \langle l \vert)`.

        :param l: Base-`d` string using `0-9A-Z` or an equivalent tensor index.
        :type l: Union[str, Tensor]
        :return: The tensornetwork nodes for the amplitude of the circuit.
        :rtype: List[Gate]
        """
        return self._circ.amplitude_before(l)

    def general_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        status: Optional[float] = None,
        with_prob: bool = False,
        name: Optional[str] = None,
    ) -> Tensor:
        """
        Monte Carlo trajectory simulation of general Kraus channel whose Kraus operators cannot be
        amplified to unitary operators. For unitary operators composed Kraus channel, :py:meth:`unitary_kraus`
        is much faster.

        This function is jittable in theory. But only jax+GPU combination is recommended for jit
        since the graph building time is too long for other backend options; though the running
        time of the function is very fast for every case.

        :param kraus: A list of ``tn.Node`` for Kraus operators.
        :type kraus: Sequence[Gate]
        :param index: The qubits index that Kraus channel is applied on.
        :type index: int
        :param status: Random tensor uniformly between 0 or 1, defaults to be None,
            when the random number will be generated automatically
        :type status: Optional[float], optional
        """
        return self._circ.general_kraus(
            kraus,
            *index,
            status=status,
            with_prob=with_prob,
            name=name,
        )

    def unitary_kraus(
        self,
        kraus: Sequence[Gate],
        *index: int,
        prob: Optional[Sequence[float]] = None,
        status: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Tensor:
        """
        Apply unitary gates in ``kraus`` randomly based on corresponding ``prob``.
        If ``prob`` is ``None``, this is reduced to kraus channel language.

        :param kraus: List of ``tc.gates.Gate`` or just Tensors
        :type kraus: Sequence[Gate]
        :param prob: prob list with the same size as ``kraus``, defaults to None
        :type prob: Optional[Sequence[float]], optional
        :param status: random seed between 0 to 1, defaults to None
        :type status: Optional[float], optional
        :return: shape [] int dtype tensor indicates which kraus gate is actually applied
        :rtype: Tensor
        """
        return self._circ.unitary_kraus(
            kraus,
            *index,
            prob=prob,
            status=status,
            name=name,
        )
