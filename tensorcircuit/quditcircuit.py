"""
Quantum circuit: the state simulator.
Supports qudit (3 <= dim <= 36) systems.
For string-encoded samples/counts, digits use 0–9A–Z where A=10, …, Z=35.
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
from .quditgates import SINGLE_BUILDERS, TWO_BUILDERS, _cached_matrix


Tensor = Any
SAMPLE_FORMAT = Literal["sample_bin", "count_dict_bin"]


class QuditCircuit:
    r"""
    ``QuditCircuit`` class.

    Qudit quick example (d=3):
    .. code-block:: python

        c = tc.QuditCircuit(2, d=3)
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
        """dimension of the qudit circuit"""
        return self._d

    @property
    def nqubits(self) -> int:
        """qudit number of the circuit"""
        return self._nqubits

    def _apply_gate(self, *indices: int, name: str, **kwargs: Any) -> None:
        """
        Apply a quantum gate (unitary) to one or two qudits in the circuit.

        The gate matrix is looked up by name in either ``SINGLE_BUILDERS`` (for single-qudit gates)
        or ``TWO_BUILDERS`` (for two-qudit gates). The matrix is built (and cached) via ``_cached_matrix``,
        then applied to the circuit at the given indices.

        :param indices: The qudit indices the gate should act on.
            - One index → single-qudit gate.
            - Two indices → two-qudit gate.
        :type indices: int
        :param name: The name of the gate (must exist in the chosen builder set).
        :type name: str
        :param kwargs: Extra parameters for the gate matched against the builder signature.
        :type kwargs: Any
        :raises ValueError: If ``name`` is not found,
        or if the number of indices does not match the gate type (single vs two).
        """
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
        """
        Apply a quantum gate (unitary) to one or two qudits in the circuit.

        :param indices: The qudit indices the gate should act on.
        :type indices: int
        :param unitary: The unitary matrix to apply to the qudit(s).
        :type unitary: Tensor
        :param name: The name to record for this gate.
        :type name: str
        """
        self._circ.unitary(*indices, unitary=unitary, name=name, dim=self._d)  # type: ignore

    unitary = any

    def i(self, index: int) -> None:
        """
        Apply the identity (I) gate on the given qudit index.

        :param index: Qudit index to apply the gate on.
        :type index: int
        """
        self._apply_gate(index, name="I")

    def x(self, index: int) -> None:
        """
        Apply the X gate on the given qudit index.

        :param index: Qudit index to apply the gate on.
        :type index: int
        """
        self._apply_gate(index, name="X")

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
        Apply the Z gate on the given qudit index.

        :param index: Qudit index to apply the gate on.
        :type index: int
        """
        self._apply_gate(index, name="Z")

    def h(self, index: int) -> None:
        """
        Apply the Hadamard-like (H) gate on the given qudit index.

        :param index: Qudit index to apply the gate on.
        :type index: int
        """
        self._apply_gate(index, name="H")

    def u8(
        self, index: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
    ) -> None:
        """
        Apply the U8 parameterized single-qudit gate.

        :param index: Qudit index to apply the gate on.
        :type index: int
        :param gamma: Gate parameter ``gamma``.
        :type gamma: float
        :param z: Gate parameter ``z``.
        :type z: float
        :param eps: Gate parameter ``eps``.
        :type eps: float
        """
        self._apply_gate(index, name="U8", extra=(gamma, z, eps))

    def rx(self, index: int, theta: float, j: int = 0, k: int = 1) -> None:
        """
        Apply the single-qudit RX rotation on ``index``.

        :param index: Qudit index to apply the gate on.
        :type index: int
        :param theta: Rotation angle.
        :type theta: float
        :param j: Source level of the rotation subspace.
        :type j: int
        :param k: Target level of the rotation subspace.
        :type k: int
        """
        self._apply_gate(index, name="RX", theta=theta, j=j, k=k)

    def ry(self, index: int, theta: float, j: int = 0, k: int = 1) -> None:
        """
        Apply the single-qudit RY rotation on ``index``.

        :param index: Qudit index to apply the gate on.
        :type index: int
        :param theta: Rotation angle.
        :type theta: float
        :param j: Source level of the rotation subspace.
        :type j: int
        :param k: Target level of the rotation subspace.
        :type k: int
        """
        self._apply_gate(index, name="RY", theta=theta, j=j, k=k)

    def rz(self, index: int, theta: float, j: int = 0) -> None:
        """
        Apply the single-qudit RZ rotation on ``index``.

        :param index: Qudit index to apply the gate on.
        :type index: int
        :param theta: Rotation angle around Z.
        :type theta: float
        :param j: Level where the phase rotation is applied.
        :type j: int
        """
        self._apply_gate(index, name="RZ", theta=theta, j=j)

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
        Apply a two-qudit RXX-type interaction on the given indices.

        :param indices: Two qudit indices.
        :type indices: int
        :param theta: Interaction strength/angle.
        :type theta: float
        :param j1: Source level of the first qudit subspace.
        :type j1: int
        :param k1: Target level of the first qudit subspace.
        :type k1: int
        :param j2: Source level of the second qudit subspace.
        :type j2: int
        :param k2: Target level of the second qudit subspace.
        :type k2: int
        """
        self._apply_gate(*indices, name="RXX", theta=theta, j1=j1, k1=k1, j2=j2, k2=k2)

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
        Apply a two-qudit RZZ-type interaction on the given indices.

        :param indices: Two qudit indices.
        :type indices: int
        :param theta: Interaction strength/angle.
        :type theta: float
        :param j1: Source level of the first qudit subspace.
        :type j1: int
        :param k1: Target level of the first qudit subspace.
        :type k1: int
        :param j2: Source level of the second qudit subspace.
        :type j2: int
        :param k2: Target level of the second qudit subspace.
        :type k2: int
        """
        self._apply_gate(*indices, name="RZZ", theta=theta, j1=j1, k1=k1, j2=j2, k2=k2)

    def cphase(self, *indices: int, cv: Optional[int] = None) -> None:
        """
        Apply a controlled phase (CPHASE) gate.

        :param indices: Two qudit indices (control, target).
        :type indices: int
        :param cv: Optional control value. If ``None``, defaults to ``1``.
        :type cv: Optional[int]
        """
        self._apply_gate(*indices, name="CPHASE", cv=cv)

    def csum(self, *indices: int, cv: Optional[int] = None) -> None:
        """
        Apply a controlled-sum (CSUM) gate.

        :param indices: Two qudit indices (control, target).
        :type indices: int
        :param cv: Optional control value. If ``None``, defaults to ``1``.
        :type cv: Optional[int]
        """
        self._apply_gate(*indices, name="CSUM", cv=cv)

    cnot = csum

    # Functional
    def wavefunction(self, form: str = "default") -> tn.Node.tensor:
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
        Returns the amplitude of the circuit given the bitstring l.
        For state simulator, it computes :math:`\langle l\vert \psi\rangle`,
        for density matrix simulator, it computes :math:`Tr(\rho \vert l\rangle \langle 1\vert)`
        Note how these two are different up to a square operation.

        :Example:

        >>> c = tc.QuditCircuit(2, dim=3)
        >>> c.x(0)
        >>> c.x(1)
        >>> c.amplitude("20")
        array(1.+0.j, dtype=complex64)
        >>> c.csum(0, 1, cv=2)
        >>> c.amplitude("21")
        array(1.+0.j, dtype=complex64)

        :param l: The bitstring of base-d characters.
        :type l: Union[str, Tensor]
        :return: The amplitude of the circuit.
        :rtype: tn.Node.tensor
        """
        return self._circ.amplitude(l)

    def probability(self) -> Tensor:
        """
        Get the ``d^n`` length probability vector over the computational basis.

        :return: Probability vector of shape ``[dim**n]``.
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
        batched sampling from state or circuit tensor network directly

        :param batch: number of samples, defaults to None
        :type batch: Optional[int], optional
        :param allow_state: if true, we sample from the final state
            if memory allows, True is preferred, defaults to False
        :type allow_state: bool, optional
        :param readout_error: readout_error, defaults to None
        :type readout_error: Optional[Sequence[Any]]. Tensor, List, Tuple
        :param format: sample format, defaults to None as backward compatibility
            check the doc in :py:meth:`tensorcircuit.quantum.measurement_results`
            Six formats of measurement counts results:

                "sample_bin": # [np.array([1, 0]), np.array([1, 0])]

                "count_vector": # np.array([2, 0, 0, 0])

                "count_dict_bin": # {"00": 2, "01": 0, "10": 0, "11": 0}
                    for cases d\in [11, 36], use 0–9A–Z digits (e.g., 'A' -> 10, …, 'Z' -> 35);

        :type format: Optional[str]
        :param random_generator: random generator,  defaults to None
        :type random_generator: Optional[Any], optional
        :param status: external randomness given by tensor uniformly from [0, 1],
            if set, can overwrite random_generator, shape [batch] for `allow_state=True`
            and shape [batch, nqubits] for `allow_state=False` using perfect sampling implementation
        :type status: Optional[Tensor]
        :param jittable: when converting to count, whether keep the full size. if false, may be conflict
            external jit, if true, may fail for large scale system with actual limited count results
        :type jittable: bool, defaults true
        :return: List (if batch) of tuple (binary configuration tensor and corresponding probability)
            if the format is None, and consistent with format when given
        :rtype: Any
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
        remaining wavefunction or density matrix on sites in ``left``, with other sites
        fixed to given digits (0..d-1) as indicated by ``traceout``

        :param traceout: can be jitted
        :type traceout: Tensor
        :param left: cannot be jitted
        :type left: Tuple
        :return: _description_
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
        Middle measurement in z-basis on the circuit, note the wavefunction output is not normalized
        with ``mid_measurement`` involved, one should normalize the state manually if needed.
        This is a post-selection method as keep is provided as a prior.

        :param index: The index of qubit that the Z direction postselection applied on.
        :type index: int
        :param keep: the post-selected digit in {0, ..., d-1}, defaults to be 0.
        :type keep: int, optional
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
        Replace the input state in MPS representation while keep the circuit structure unchanged.

        :Example:
        >>> c = tc.QuditCircuit(2, dim=3)
        >>> c.x(0)
        >>>
        >>> c2 = tc.QuditCircuit(2, dim=3, mps_inputs=c.quvector())
        >>> c2.x(0)
        >>> c2.wavefunction()
        array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)
        >>>
        >>> c3 = tc.QuditCircuit(2, dim=3)
        >>> c3.x(0)
        >>> c3.replace_mps_inputs(c.quvector())
        >>> c3.wavefunction()
        array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)

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
        Get the tensor network in the form of a list of nodes
        for the expectation calculation before the real contraction

        :param reuse: _description_, defaults to True
        :type reuse: bool, optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: List[tn.Node]
        """
        return self._circ.expectation_before(*ops, reuse=reuse, **kws)

    def amplitude_before(self, l: Union[str, Tensor]) -> List[Gate]:
        r"""
        Returns the tensornetwor nodes for the amplitude of the circuit given the bitstring l.
        For state simulator, it computes :math:`\langle l\vert \psi\rangle`,
        for density matrix simulator, it computes :math:`Tr(\rho \vert l\rangle \langle 1\vert)`
        Note how these two are different up to a square operation.

        :param l: The bitstring of 0 and 1s.
        :type l: Union[str, Tensor]
        :return: The tensornetwork nodes for the amplitude of the circuit.
        :rtype: List[Gate]
        """
        return self._circ.amplitude_before(l)