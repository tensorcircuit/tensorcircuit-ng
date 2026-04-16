"""
SymbolCircuit: symbolic parameterized quantum circuit.

Gate parameters are sympy Symbols (or expressions). Amplitude and expectation
values are computed via tensor network contraction of numpy object arrays,
producing sympy expressions. The class also supports translation to a Qiskit
QuantumCircuit with Parameter objects for hardware compilation reuse.

Key design: inherit from Circuit, override gate registration to use symbolic
factories from symbolic_gates.py instead of the standard backend-coupled ones,
and override the handful of computation methods that call backend.* on tensor
values.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import logging
import math

import numpy as np
import sympy
import tensornetwork as tn

from .circuit import Circuit
from .cons import contractor, runtime_backend, set_function_backend
from .gates import Gate
from .quantum import QuOperator, _decode_basis_label
from .symbolic_gates import SYM_SGATE_MAP, SYM_VGATE_MAP, sym_r
from .utils import is_sequence
from .simplify import _full_light_cone_cancel

logger = logging.getLogger(__name__)

Tensor = Any

backend_numpy = set_function_backend("numpy")


class SymbolCircuit(Circuit):
    """
    Quantum circuit with symbolic (sympy) gate parameters.

    Gate parameters are ``sympy.Symbol`` objects (or sympy expressions).
    Amplitude and expectation values return sympy expressions via tensor network
    contraction. The circuit can be translated to a Qiskit
    ``QuantumCircuit`` with ``Parameter`` objects for hardware reuse.

    **Backend isolation** — ``SymbolCircuit`` is permanently fixed to the numpy
    backend regardless of any global backend setting.  Calling
    ``tc.set_backend("jax")`` (or ``"tensorflow"``, ``"torch"``, etc.) before or
    after constructing a ``SymbolCircuit`` has *no effect* on its internal
    computation.  This is by design: the class represents all state vectors and
    gate matrices as ``numpy.ndarray`` with ``dtype=object``, whose entries are
    sympy expressions.  The methods ``amplitude``, ``wavefunction``, and
    ``expectation`` / ``expectation_before`` are all overridden to use plain
    NumPy operations instead of ``tc.backend.*`` calls, so they never touch the
    active backend.

    The isolation ends at :meth:`to_circuit`: the returned :class:`Circuit` is a
    standard numerical circuit that *does* respect the global backend setting at
    the time it is called.

    Example::

        import sympy
        import tensorcircuit as tc

        theta = sympy.Symbol("theta", real=True)

        sc = tc.SymbolCircuit(2)
        sc.h(0)
        sc.rx(1, theta=theta)
        sc.cnot(0, 1)

        # symbolic expectation — always numpy / sympy, unaffected by set_backend
        expr = sc.expectation_ps(z=[0, 1])
        print(sympy.simplify(expr))

        # bind symbols → standard Circuit that uses the active backend
        c = sc.to_circuit({theta: 0.5})

        # Qiskit PQC for hardware
        qc = sc.to_qiskit()
        print(qc.parameters)
    """

    is_dm = False
    is_mps = False

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        nqubits: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[Any] = None,
        tensors: Optional[Any] = None,
        split: Optional[Dict[str, Any]] = None,
        dim: Optional[int] = None,
    ) -> None:
        """
        Initialize a SymbolCircuit with ``nqubits`` qubits.

        The initial state is :math:`|0\\rangle^{\\otimes n}` represented as
        numpy object-dtype tensor network nodes (compatible with sympy).

        :param nqubits: Number of qubits.
        :type nqubits: int
        """
        if dim is not None and dim != 2:
            raise ValueError("SymbolCircuit only supports qubit (dim=2) for now.")
        if mps_inputs is not None or tensors is not None:
            raise NotImplementedError(
                "mps_inputs and tensors are not yet supported in SymbolCircuit"
            )

        self._nqubits = nqubits
        self._d = 2
        self.split = split
        self.inputs = inputs

        self.circuit_param = {
            "nqubits": nqubits,
            "inputs": inputs,
            "split": split,
            "dim": dim,
        }

        with runtime_backend("numpy"):
            if inputs is None:
                # Create |0> nodes as object-dtype numpy arrays — all backend-agnostic
                nodes = []
                for _ in range(nqubits):
                    node = tn.Node(np.array([1, 0], dtype=object), name="qb")
                    self.coloring_nodes([node], flag="inputs")
                    nodes.append(node)

                self._front: List[tn.Edge] = [n.get_edge(0) for n in nodes]
            else:
                inputs = np.asarray(inputs, dtype=object)
                inputs = inputs.reshape([self._d] * nqubits)
                inputs_node = Gate(inputs)
                self.coloring_nodes([inputs_node], flag="inputs")
                nodes = [inputs_node]
                self._front = [inputs_node.get_edge(i) for i in range(nqubits)]

        self._nodes: List[tn.Node] = nodes
        self._start_index = len(nodes)

        self._qir: List[Dict[str, Any]] = []
        self._extra_qir: List[Dict[str, Any]] = []
        self._measure_counter = 0

    # ── gate registration overrides ───────────────────────────────────────────

    @staticmethod
    def apply_general_gate_delayed(
        gatef: Callable[..., Any],
        name: Optional[str] = None,
        mpo: bool = False,
    ) -> Callable[..., None]:
        """
        Override for fixed gates: use symbolic gate factory instead of
        the backend-coupled ``gatef()`` call.
        """
        if name is None:
            name = getattr(gatef, "n")
        defaultname = name

        @backend_numpy
        def apply(
            self: "SymbolCircuit",
            *index: int,
            split: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
        ) -> None:
            localname = name if name is not None else defaultname
            sym_gatef = SYM_SGATE_MAP.get(localname)  # type: ignore
            if sym_gatef is None:
                sym_gatef = SYM_SGATE_MAP.get(defaultname)
            if sym_gatef is not None:
                gate = sym_gatef()
            else:
                gate = gatef()
            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                ir_dict={"gatef": gatef},
            )

        def apply_list(self: "SymbolCircuit", *index: int, **kws: Any) -> None:
            if isinstance(index[0], int):
                apply(self, *index, **kws)
            elif is_sequence(index[0]) or isinstance(index[0], range):
                for ind in zip(*index):
                    apply(self, *ind, **kws)
            else:
                raise ValueError("Illegal index specification")

        return apply_list

    @staticmethod
    def apply_general_variable_gate_delayed(
        gatef: Callable[..., Any],
        name: Optional[str] = None,
        mpo: bool = False,
        diagonal: bool = False,
    ) -> Callable[..., None]:
        """
        Override for variable gates: use symbolic gate factory (sympy cos/sin)
        instead of the backend-coupled ``gatef(**vars)`` call.
        """
        if name is None:
            name = getattr(gatef, "n")

        @backend_numpy
        def apply(self: "SymbolCircuit", *index: int, **vars: Any) -> None:
            split = vars.pop("split", None)
            localname = vars.pop("name", name)

            sym_gatef = SYM_VGATE_MAP.get(localname)  # type: ignore
            if sym_gatef is None:
                sym_gatef = SYM_VGATE_MAP.get(name)  # type: ignore
            if sym_gatef is not None:
                gate = sym_gatef(**vars)  # type: ignore[operator]
            else:
                raise NotImplementedError(
                    f"Symbolic gate '{name}' is not yet supported. "
                    f"Supported variable gates: {list(SYM_VGATE_MAP)}"
                )

            self.apply_general_gate(
                gate,
                *index,
                name=localname,
                split=split,
                mpo=mpo,
                diagonal=diagonal,
                ir_dict={
                    "gatef": gatef,
                    "index": index,
                    "name": localname,
                    "split": split,
                    "mpo": mpo,
                    "diagonal": diagonal,
                    "parameters": dict(vars),
                },
            )

        def apply_list(self: "SymbolCircuit", *index: int, **vars: Any) -> None:
            if isinstance(index[0], int):
                apply(self, *index, **vars)
            elif is_sequence(index[0]) or isinstance(index[0], range):
                for i, ind in enumerate(zip(*index)):
                    nvars: Dict[str, Any] = {}
                    for k, v in vars.items():
                        try:
                            nvars[k] = v[i]
                        except Exception:  # pylint: disable=W0703
                            nvars[k] = v
                    apply(self, *ind, **nvars)
            else:
                raise ValueError("Illegal index specification")

        return apply_list

    @backend_numpy
    def apply_general_gate(
        self,
        gate: Union[Gate, QuOperator],
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        diagonal: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Override for backend isolation.
        """
        super().apply_general_gate(
            gate,
            *index,
            name=name,
            split=split,
            mpo=mpo,
            diagonal=diagonal,
            ir_dict=ir_dict,
        )

    @backend_numpy
    def _copy(
        self, conj: Optional[bool] = False
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        Override for backend isolation.
        """
        return super()._copy(conj=conj)

    @backend_numpy
    def _copy_state_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        Override for backend isolation.
        """
        return super()._copy_state_tensor(conj=conj, reuse=reuse)

    # ── amplitude ──────────────────────────────────────────────────────────────

    @backend_numpy
    def amplitude(self, l: Union[str, Sequence[int]]) -> Any:
        r"""
        Compute :math:`\langle l \vert \psi \rangle` symbolically.

        :param l: Bitstring as a string (e.g. ``"01"``) or sequence of ints.
        :type l: Union[str, Sequence[int]]
        :return: Sympy expression for the amplitude.
        :rtype: sympy expression
        """
        no, d_edges = self._copy()

        if isinstance(l, str):
            bits = _decode_basis_label(l, n=self._nqubits, dim=self._d)
        else:
            bits = list(l)

        ms = []
        for i, bit in enumerate(bits):
            bra = np.zeros(self._d, dtype=object)
            bra[int(bit)] = 1
            n = tn.Node(bra)
            self.coloring_nodes([n], flag="measurement")
            ms.append(n)
            d_edges[i] ^ n.get_edge(0)
        no.extend(ms)

        result = contractor(no).tensor
        if hasattr(result, "item"):
            result = result.item()
        return result

    # ── wavefunction / state ──────────────────────────────────────────────────

    @backend_numpy
    def wavefunction(self, form: str = "default") -> np.ndarray:  # type: ignore[type-arg]
        """
        Compute the symbolic output state vector.

        Returns a numpy object array containing sympy expressions. Only
        practical for small qubit counts where the full vector is manageable.

        :param form: Shape of output: ``"default"`` → 1-D, ``"ket"`` → column,
            ``"bra"`` → row.  Defaults to ``"default"``.
        :type form: str, optional
        :return: Numpy object array of sympy expressions.
        :rtype: np.ndarray
        """
        nodes, d_edges = self._copy()
        t = contractor(nodes, output_edge_order=d_edges)
        arr = t.tensor.reshape(-1)
        if form == "ket":
            arr = arr.reshape(-1, 1)
        elif form == "bra":
            arr = arr.reshape(1, -1)
        return arr  # type: ignore[no-any-return]

    state = wavefunction

    @backend_numpy
    def matrix(self) -> np.ndarray:  # type: ignore[type-arg]
        """
        Compute the symbolic unitary matrix of the circuit.
        """
        return super().matrix()  # type: ignore[no-any-return]

    @backend_numpy
    def probability(self) -> np.ndarray:  # type: ignore[type-arg]
        """
        Compute the symbolic probability distribution.
        """
        psi = self.wavefunction()
        return np.array([sympy.Abs(amp) ** 2 for amp in psi], dtype=object)

    @backend_numpy
    def get_quoperator(self) -> QuOperator:
        """
        Get symbolic QuOperator representation.
        """
        return super().get_quoperator()

    quoperator = get_quoperator

    @backend_numpy
    def projected_subsystem(self, traceout: Any, left: Tuple[int, ...]) -> Any:
        """
        Compute symbolic projected subsystem.
        """
        # Ensure traceout is concrete (not symbolic) for bit selection
        try:
            traceout = [int(v) for v in traceout]
        except (TypeError, AttributeError):
            raise ValueError(
                "Symbolic projected_subsystem requires concrete integer-convertible "
                "basis states in `traceout`."
            ) from None

        all_endns = []
        for v in traceout:
            vec = np.zeros(self._d, dtype=object)
            vec[v] = 1
            all_endns.append(vec)

        nodes, front = self._copy()
        # L = self._nqubits
        edges = []
        for i in range(len(traceout)):
            if i not in left:
                n = Gate(all_endns[i])
                nodes.append(n)
                front[i] ^ n.get_edge(0)
            else:
                edges.append(front[i])

        t = contractor(nodes, output_edge_order=edges)
        # backend.norm fails on sympy object array, we do it manually symbolically
        raw_tensor = t.tensor
        norm_sq = np.sum([sympy.Abs(x) ** 2 for x in raw_tensor.reshape(-1)])
        if norm_sq == 0:
            raise ValueError(
                "Projected subsystem has zero norm (impossible projection)."
            )
        norm = sympy.sqrt(norm_sq)
        return (raw_tensor / norm).reshape(-1)

    # ── expectation ────────────────────────────────────────────────────────────

    @backend_numpy
    def expectation_before(
        self,
        *ops: Tuple[Any, List[int]],
        reuse: bool = True,
        **kws: Any,
    ) -> List[tn.Node]:
        """
        Build the tensor network for ``<psi|O|psi>`` without contracting.

        Operators may be:
        * A ``Gate`` / ``tn.Node`` (numerical or symbolic tensor)
        * A plain ``np.ndarray``

        All operator tensors are converted to numpy ``dtype=object`` for
        compatibility with the symbolic state tensor.
        """
        nq = self._nqubits
        nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
        nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
        nodes = nodes1 + nodes2
        newdang = edge1 + edge2

        occupied: Set[int] = set()
        for op, index in ops:
            # Normalise operator to Gate with object-dtype tensor
            if not isinstance(op, tn.Node):
                op_arr = np.asarray(op, dtype=object)
                n_legs = int(round(np.log2(op_arr.size)))
                op = Gate(op_arr.reshape([2] * n_legs))
            else:
                if op.tensor.dtype != object:
                    op = Gate(np.array(op.tensor, dtype=object))
                    # reshape is already correct for Gate

            if isinstance(index, int):
                index = [index]
            index = tuple(i if i >= 0 else nq + i for i in index)  # type: ignore[assignment]
            noe = len(index)

            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                newdang[e + nq] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            self.coloring_nodes([op], flag="operator")
            nodes.append(op)

        for j in range(nq):
            if j not in occupied:
                newdang[j] ^ newdang[j + nq]

        return nodes  # type: ignore[no-any-return]

    @backend_numpy
    def expectation(
        self,
        *ops: Tuple[Any, List[int]],
        reuse: bool = True,
        enable_lightcone: bool = False,
        **kws: Any,
    ) -> Any:
        r"""
        Compute :math:`\langle \psi | O | \psi \rangle` symbolically.

        :param ops: Pairs of ``(operator, qubit_indices)``.  The operator may
            be a ``Gate``, a ``tn.Node``, or a plain numpy array.  Symbolic
            (object-dtype) operators are supported.
        :type ops: Tuple[operator, List[int]]
        :param reuse: Cache the contracted state vector for repeated calls,
            defaults to True.
        :type reuse: bool
        :param enable_lightcone: whether enable light cone simplification, defaults to False
        :type enable_lightcone: bool
        :return: Sympy expression for the expectation value.
        """
        if enable_lightcone:
            reuse = False
        nodes = self.expectation_before(*ops, reuse=reuse)
        if enable_lightcone:
            nodes = _full_light_cone_cancel(nodes)
        result = contractor(nodes).tensor
        if hasattr(result, "item"):
            result = result.item()
        return result

    @backend_numpy
    def sample_expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        shots: Optional[int] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Any] = None,
        readout_error: Optional[Sequence[Any]] = None,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        statusc: Optional[Any] = None,
        **kws: Any,
    ) -> Any:
        """
        Symbolic execution for analytical Pauli string expectation.

        :param x: List of indices for Pauli X.
        :type x: Optional[Sequence[int]]
        :param y: List of indices for Pauli Y.
        :type y: Optional[Sequence[int]]
        :param z: List of indices for Pauli Z.
        :type z: Optional[Sequence[int]]
        :param shots: Must be None for analytical symbolic result.
        :type shots: Optional[int]
        :return: Sympy expression for the expectation value.
        """
        if shots is not None:
            raise NotImplementedError(
                "SymbolCircuit does not support numerical sampling. "
                "Use sc.to_circuit() to convert to a numerical circuit first."
            )
        if (
            random_generator is not None
            or status is not None
            or readout_error is not None
            or noise_conf is not None
            or statusc is not None
        ):
            raise NotImplementedError(
                "SymbolCircuit analytical expectation does not support noise or custom "
                "sampling parameters."
            )

        # Analytical (symbolic) expectation
        return self.expectation_ps(x=x, y=y, z=z, **kws)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overridden to provide better error message for symbolic circuits.
        """
        try:
            return self.to_circuit().sample(*args, **kwargs)
        except ValueError as e:
            if "free symbols" in str(e):
                raise NotImplementedError(
                    "SymbolCircuit.sample is only supported when all symbols are bound. "
                    "Bind symbols first using sc.to_circuit(param_dict)."
                ) from None
            raise

    def measure(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overridden to provide better error message for symbolic circuits.
        """
        try:
            return self.to_circuit().measure(*args, **kwargs)
        except ValueError as e:
            if "free symbols" in str(e):
                raise NotImplementedError(
                    "SymbolCircuit.measure is only supported when all symbols are bound. "
                    "Bind symbols first using sc.to_circuit(param_dict)."
                ) from None
            raise

    def measure_reference(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overridden to provide better error message for symbolic circuits.
        """
        try:
            return self.to_circuit().measure_reference(*args, **kwargs)
        except ValueError as e:
            if "free symbols" in str(e):
                raise NotImplementedError(
                    "SymbolCircuit.measure_reference is only supported when all symbols are bound. "
                    "Bind symbols first using sc.to_circuit(param_dict)."
                ) from None
            raise

    def cond_measurement(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overridden to provide better error message for symbolic circuits.
        """
        raise NotImplementedError(
            "SymbolCircuit does not support cond_measurement (Monte Carlo). "
            "Convert to a numerical circuit using sc.to_circuit() first."
        )

    cond_measure = cond_measurement

    # ── symbol utilities ──────────────────────────────────────────────────────

    def free_symbols(self) -> Set[sympy.Symbol]:
        """
        Return the set of all free sympy Symbols used as gate parameters.

        :return: Set of sympy Symbols.
        :rtype: Set[sympy.Symbol]
        """
        syms: Set[sympy.Symbol] = set()
        for d in self._qir:
            for v in d.get("parameters", {}).values():
                if hasattr(v, "free_symbols"):
                    syms |= v.free_symbols
        return syms

    def _bind_scalar(self, k: str, v: Any, param_dict: Dict[sympy.Symbol, Any]) -> Any:
        if hasattr(v, "subs"):
            v = v.subs(param_dict)
        if hasattr(v, "free_symbols") and v.free_symbols:
            raise ValueError(
                f"Parameter '{k}' still contains free symbols "
                f"{v.free_symbols} after substitution. "
                "Pass a complete param_dict to to_circuit()."
            )
        return complex(v) if hasattr(v, "free_symbols") else v

    def _bind_array(
        self, k: str, v: Any, param_dict: Dict[sympy.Symbol, Any]
    ) -> np.ndarray:  # type: ignore[type-arg]
        arr = np.asarray(v)
        if arr.dtype != object:
            return arr
        bound = np.vectorize(lambda x: self._bind_scalar(k, x, param_dict))(arr)
        return bound.astype(complex)  # type: ignore[no-any-return]

    def to_circuit(
        self, param_dict: Optional[Dict[sympy.Symbol, Any]] = None
    ) -> Circuit:
        """
        Convert to a numerical :class:`Circuit` by binding symbolic parameters.

        :param param_dict: Mapping from sympy Symbol to numerical value.
            Pass ``None`` (or ``{}``) only if the circuit has no free symbols.
        :type param_dict: Optional[Dict[sympy.Symbol, Any]]
        :return: A fully numerical :class:`Circuit`.
        :rtype: Circuit
        """
        if param_dict is None:
            param_dict = {}
        c = Circuit(self._nqubits)
        for d in self._qir:
            gate_name = d["name"]
            index = d["index"]
            params = {
                k: (
                    self._bind_array(k, v, param_dict)
                    if isinstance(v, (np.ndarray, list))
                    else self._bind_scalar(k, v, param_dict)
                )
                for k, v in d.get("parameters", {}).items()
            }
            getattr(c, gate_name)(*index, **params)
        return c

    def bind(self, param_dict: Dict[sympy.Symbol, Any]) -> "SymbolCircuit":
        """
        Return a new :class:`SymbolCircuit` with some or all parameters
        substituted (partial or full binding).

        :param param_dict: Mapping from sympy Symbol to value (numeric or
            another sympy expression).
        :type param_dict: Dict[sympy.Symbol, Any]
        :return: New :class:`SymbolCircuit` with substituted parameters.
        :rtype: SymbolCircuit
        """
        sc = SymbolCircuit(self._nqubits)
        for d in self._qir:
            gate_name = d["name"]
            index = d["index"]
            params = {}
            for k, v in d.get("parameters", {}).items():
                if hasattr(v, "subs"):
                    v = sympy.simplify(v.subs(param_dict))
                params[k] = v
            if params:
                getattr(sc, gate_name)(*index, **params)
            else:
                getattr(sc, gate_name)(*index)
        return sc

    # ── Qiskit translation ────────────────────────────────────────────────────

    def to_qiskit(
        self,
        enable_instruction: bool = False,
        enable_inputs: bool = False,
    ) -> Any:
        """
        Translate to a Qiskit ``QuantumCircuit`` with ``Parameter`` objects.

        Each ``sympy.Symbol`` used as a gate parameter is mapped to a Qiskit
        ``Parameter`` with the same name.  The resulting circuit can be bound
        with :meth:`qiskit.QuantumCircuit.assign_parameters` and executed on
        hardware or simulators.

        Simple arithmetic expressions involving symbols are translated when
        possible (``+``, ``-``, ``*``, ``/``).  Complex expressions (``sin``,
        ``cos``, etc.) that appear directly as *gate-level parameters* will
        raise a ``NotImplementedError`` — but note that rotation angles are
        passed at the gate level (e.g. ``rx(theta)``), not as matrix entries,
        so this is rarely an issue for standard circuit translation.

        :return: Qiskit ``QuantumCircuit`` with symbolic Parameters.
        :rtype: qiskit.circuit.QuantumCircuit
        """
        try:
            from qiskit.circuit import Parameter, QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for to_qiskit(); install with `pip install qiskit`."
            ) from exc

        qc = QuantumCircuit(self._nqubits)
        sym_to_qk: Dict[sympy.Symbol, Parameter] = {}

        def _to_qk(expr: Any) -> Any:
            """Convert a sympy expression to a Qiskit parameter or float."""
            if isinstance(expr, (int, float, complex)):
                return float(expr)  # type: ignore[arg-type]
            if isinstance(expr, sympy.Number):
                val = complex(expr)
                return float(val.real) if val.imag == 0 else val
            if isinstance(expr, sympy.Symbol):
                if expr not in sym_to_qk:
                    sym_to_qk[expr] = Parameter(str(expr))
                return sym_to_qk[expr]
            # Attempt to handle simple arithmetic on symbols
            if isinstance(expr, sympy.Expr) and expr.free_symbols:
                # Ensure all symbols are registered
                for sym in expr.free_symbols:
                    if sym not in sym_to_qk:
                        sym_to_qk[sym] = Parameter(str(sym))
                # Try sympy → Qiskit ParameterExpression via rebuild
                return _sym_expr_to_qk(expr, sym_to_qk)
            # Numeric (no free symbols)
            val = complex(expr)
            return float(val.real) if val.imag == 0 else val

        for d in self._qir:
            if d.get("is_channel"):
                continue
            gate_name = d["name"]
            index = list(d["index"])
            params = d.get("parameters", {})

            if gate_name in (
                "i",
                "x",
                "y",
                "z",
                "h",
                "t",
                "s",
                "cnot",
                "cx",
                "cz",
                "cy",
                "swap",
                "toffoli",
                "ccnot",
                "ccx",
                "fredkin",
                "cswap",
            ):
                getattr(qc, gate_name)(*index)
            elif gate_name == "wroot":
                raise NotImplementedError(
                    "SymbolCircuit.to_qiskit: wroot has no native Qiskit equivalent. "
                    "Decompose the gate or bind all parameters and use a UnitaryGate."
                )
            elif gate_name in ("sd", "td"):
                getattr(qc, gate_name + "g")(*index)
            elif gate_name in ("ox", "oy", "oz"):
                getattr(qc, "c" + gate_name[1:])(*index, ctrl_state=0)
            elif gate_name in ("orx", "ory", "orz"):
                # Open-controlled rotation: ctrl_state=0 means control fires on |0>
                getattr(qc, "c" + gate_name[1:])(
                    _to_qk(params.get("theta", 0)), *index, ctrl_state=0
                )
            elif gate_name in (
                "rx",
                "ry",
                "rz",
                "crx",
                "cry",
                "crz",
                "rxx",
                "ryy",
                "rzz",
            ):
                getattr(qc, gate_name)(_to_qk(params.get("theta", 0)), *index)
            elif gate_name in ("phase", "cphase"):
                qk_name = "p" if gate_name == "phase" else "cp"
                getattr(qc, qk_name)(_to_qk(params.get("theta", 0)), *index)
            elif gate_name == "iswap":
                from qiskit.circuit.library import XXPlusYYGate

                theta_qk = _to_qk(params.get("theta", 1))
                qc.append(XXPlusYYGate(math.pi * theta_qk, math.pi), index)
            elif gate_name == "u":
                qc.u(
                    _to_qk(params.get("theta", 0)),
                    _to_qk(params.get("phi", 0)),
                    _to_qk(params.get("lbd", 0)),
                    *index,
                )
            elif gate_name == "cu":
                qc.cu(
                    _to_qk(params.get("theta", 0)),
                    _to_qk(params.get("phi", 0)),
                    _to_qk(params.get("lbd", 0)),
                    0,
                    *index,
                )
            elif gate_name == "r":
                logger.warning(
                    "SymbolCircuit.to_qiskit: r gate converted via unitary (only for numeric params). "
                    "Symbolic r gate parameters may not translate correctly to Qiskit."
                )
                try:
                    num_params = {k: complex(v) for k, v in params.items()}
                except (TypeError, ValueError) as exc:
                    raise NotImplementedError(
                        "SymbolCircuit.to_qiskit: r gate with symbolic parameters cannot "
                        "be translated to Qiskit. Bind all parameters first with to_circuit() "
                        "or bind()."
                    ) from exc
                m = sym_r(**num_params).tensor.reshape(2, 2)
                m_num = np.array(m.tolist(), dtype=complex)
                from qiskit.extensions import UnitaryGate

                qc.append(UnitaryGate(m_num), index)
            else:
                # Fallback: try direct method call (for gate aliases not listed above)
                try:
                    if params:
                        num_params = {k: _to_qk(v) for k, v in params.items()}
                        getattr(qc, gate_name)(*list(num_params.values()), *index)
                    else:
                        getattr(qc, gate_name)(*index)
                except AttributeError:
                    logger.warning(
                        f"SymbolCircuit.to_qiskit: skipping unsupported gate '{gate_name}'"
                    )

        return qc


def _sym_expr_to_qk(expr: sympy.Expr, sym_to_qk: Dict[sympy.Symbol, Any]) -> Any:
    """
    Recursively translate a sympy expression to a Qiskit ParameterExpression.

    Only ``Add``, ``Mul``, and ``Pow`` with integer exponent are supported.
    For unsupported expression types, the function falls back to evaluating
    the expression numerically (which will fail if there are free symbols).
    """
    if isinstance(expr, sympy.Symbol):
        return sym_to_qk[expr]
    if isinstance(expr, sympy.Number):
        return complex(expr)
    if isinstance(expr, sympy.Add):
        parts = [_sym_expr_to_qk(a, sym_to_qk) for a in expr.args]
        result = parts[0]
        for p in parts[1:]:
            result = result + p
        return result
    if isinstance(expr, sympy.Mul):
        parts = [_sym_expr_to_qk(a, sym_to_qk) for a in expr.args]
        result = parts[0]
        for p in parts[1:]:
            result = result * p
        return result
    if isinstance(expr, sympy.Pow) and expr.exp.is_integer:
        base = _sym_expr_to_qk(expr.base, sym_to_qk)
        exp = int(expr.exp)
        result = base
        for _ in range(abs(exp) - 1):
            result = result * base
        if exp < 0:
            result = 1 / result
        return result
    # Non-translatable: try numeric evaluation
    try:
        val = complex(expr)
        return float(val.real) if val.imag == 0 else val
    except TypeError as exc:
        raise NotImplementedError(
            f"Cannot translate sympy expression '{expr}' to a Qiskit ParameterExpression. "
            f"Expression type '{type(expr).__name__}' is not supported."
        ) from exc


# Register all gate methods on SymbolCircuit via the overridden static methods
SymbolCircuit._meta_apply()
