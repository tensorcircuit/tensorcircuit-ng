"""
Host-side circuit routing to a nearest-neighbor qubit chain.

Layouts map logical qubits of the input circuit to physical positions. The
router retains its final permutation instead of restoring the initial order.
Gate tensors, including fused matrices, stay on the active backend.
"""

from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .. import gates
from ..circuit import Circuit
from ..cons import backend
from ..symbolcircuit import SymbolCircuit

Instruction = Dict[str, Any]
Pair = Tuple[int, int]

_DIAGONAL_GATES = {
    "z",
    "s",
    "sd",
    "t",
    "td",
    "rz",
    "phase",
    "cz",
    "cphase",
    "crz",
    "oz",
    "orz",
    "rzz",
}


def _layout(layout: Sequence[int], n: int) -> List[int]:
    if (
        len(layout) != n
        or any(not isinstance(p, Integral) for p in layout)
        or set(layout) != set(range(n))
    ):
        raise ValueError("layout must be a permutation of range(nqubits)")
    return [int(p) for p in layout]


def _dense_instruction(matrix: Any, index: Sequence[int]) -> Instruction:
    return {
        "gatef": gates.any,  # type: ignore[attr-defined]
        "parameters": {"unitary": matrix},
        "index": tuple(index),
        "name": "any",
        "split": None,
        "mpo": False,
        "diagonal": False,
    }


def _matrix(instruction: Instruction) -> Any:
    """Return a two-qubit matrix in ascending physical-site order for fusion."""
    if "gate" in instruction:
        tensor = instruction["gate"].tensor
    else:
        tensor = instruction["gatef"](**instruction.get("parameters", {})).tensor
    if instruction["index"][0] > instruction["index"][1]:
        tensor = backend.transpose(backend.reshape(tensor, [2] * 4), [1, 0, 3, 2])
    return backend.reshapem(tensor)


def _prepare(circuit: Circuit) -> Tuple[List[Instruction], Set[Pair], Set[int]]:
    if not isinstance(circuit, Circuit) or isinstance(circuit, SymbolCircuit):
        raise TypeError("linear_route requires a Circuit")
    if circuit._d != 2:
        raise NotImplementedError("linear_route supports qubits only")
    if any(
        circuit.circuit_param.get(key) is not None
        for key in ("inputs", "mps_inputs", "tensors")
    ):
        raise NotImplementedError(
            "linear_route requires the default product input state"
        )
    if circuit._extra_qir:
        raise NotImplementedError(
            "linear_route does not support extra QIR instructions"
        )
    qir, edges, diagonal = [], set(), set()
    for origin, source in enumerate(circuit.to_qir()):
        if source.get("is_channel", False):
            raise NotImplementedError(
                "linear_route does not support channel instructions"
            )
        if source.get("mpo", False):
            raise NotImplementedError("linear_route does not support MPO gates")
        index = tuple(
            int(q) if q >= 0 else circuit._nqubits + int(q) for q in source["index"]
        )
        if len(index) not in (1, 2):
            raise NotImplementedError("linear_route supports one- and two-qubit gates")
        instruction = dict(source)
        instruction["index"] = index
        if source.get("diagonal", False):
            if source["gatef"] is not gates.diagonal_gate:
                raise NotImplementedError("unsupported diagonal hyperedge gate")
            instruction = _dense_instruction(
                backend.diagflat(backend.reshape(source["gate"].tensor, [-1])), index
            )
            diagonal.add(origin)
        elif getattr(source["gatef"], "n", "") in _DIAGONAL_GATES:
            diagonal.add(origin)
        qir.append(instruction)
        if len(index) == 2:
            edges.add((min(index), max(index)))
    return qir, edges, diagonal


def _widths(edges: Set[Pair], layout: Sequence[int]) -> Tuple[int, int]:
    cuts = [0] * (len(layout) + 1)
    bandwidth = 0
    for i, j in edges:
        a, b = sorted((layout[i], layout[j]))
        bandwidth = max(bandwidth, b - a)
        cuts[a + 1] += 1
        cuts[b + 1] -= 1
    active, cutwidth = 0, 0
    for delta in cuts:
        active += delta
        cutwidth = max(cutwidth, active)
    return bandwidth, cutwidth


def _initial_layout(n: int, edges: Set[Pair]) -> List[int]:
    identity = list(range(n))
    if not edges:
        return identity
    pairs = sorted(edges)
    rows = [i for i, _ in pairs] + [j for _, j in pairs]
    cols = [j for _, j in pairs] + [i for i, _ in pairs]
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    order = reverse_cuthill_mckee(graph, symmetric_mode=True)
    layout = [int(p) for p in np.argsort(order)]
    return (
        layout if _widths(edges, layout)[0] < _widths(edges, identity)[0] else identity
    )


class _Router:
    def __init__(self, layout: Sequence[int], fuse: bool):
        self.pos = list(layout)
        self.at = sorted(range(len(layout)), key=self.pos.__getitem__)
        self.fuse = fuse
        self.qir: List[Instruction] = []
        self.origins: List[Tuple[int, ...]] = []
        self.last: List[Optional[int]] = [None] * len(layout)
        self.center = 0
        self.num_swaps = 0

    def _fusible(self, p: int) -> bool:
        previous = self.last[p]
        return (
            self.fuse
            and previous is not None
            and previous == self.last[p + 1]
            and len(self.qir[previous]["index"]) == 2
        )

    def _emit(self, instruction: Instruction, origins: Tuple[int, ...]) -> None:
        index = instruction["index"]
        if len(index) == 2:
            p, q = sorted(index)
            previous = self.last[p]
            if previous is not None and self._fusible(p):
                self.qir[previous] = _dense_instruction(
                    backend.matmul(_matrix(instruction), _matrix(self.qir[previous])),
                    (p, q),
                )
                self.origins[previous] = tuple(sorted(self.origins[previous] + origins))
                return
            self.center = p if abs(self.center - p) < abs(self.center - q) else q
        else:
            self.center = index[0]
        self.qir.append(instruction)
        self.origins.append(origins)
        for p in instruction["index"]:
            self.last[p] = len(self.qir) - 1

    def gate(self, instruction: Instruction, origin: int) -> None:
        mapped = dict(instruction)
        mapped["index"] = tuple(self.pos[q] for q in instruction["index"])
        self._emit(mapped, (origin,))

    def swap(self, p: int) -> None:
        self._emit(
            {
                "gatef": gates.swap,  # type: ignore[attr-defined]
                "index": (p, p + 1),
                "name": "swap",
                "split": None,
                "mpo": False,
            },
            (),
        )
        a, b = self.at[p], self.at[p + 1]
        self.at[p], self.at[p + 1] = b, a
        self.pos[a], self.pos[b] = p + 1, p
        self.num_swaps += 1

    def ordinary(self, instruction: Instruction, origin: int) -> None:
        index = instruction["index"]
        if len(index) == 2:
            left, right = sorted(self.pos[q] for q in index)
            if abs(self.center - left) < abs(self.center - right):
                for p in range(left, right - 1):
                    self.swap(p)
            else:
                for p in range(right - 1, left, -1):
                    self.swap(p)
        self.gate(instruction, origin)

    def _ready(self, pending: Dict[int, Instruction]) -> None:
        ready = []
        for origin, instruction in pending.items():
            positions = [self.pos[q] for q in instruction["index"]]
            if len(positions) == 1 or abs(positions[0] - positions[1]) == 1:
                ready.append((min(positions), origin))
        if ready:
            low, high = min(ready)[0], max(ready)[0]
            reverse = abs(self.center - high) < abs(self.center - low)
            ready.sort(key=lambda item: (-item[0] if reverse else item[0], item[1]))
            for _, origin in ready:
                self.gate(pending.pop(origin), origin)

    def _greedy_swap(self, pending: Dict[int, Instruction]) -> Optional[int]:
        balance = [0] * len(self.at)
        for instruction in pending.values():
            i, j = instruction["index"]
            sign = 1 if self.pos[i] > self.pos[j] else -1
            balance[i] += sign
            balance[j] -= sign
        best = (0, 2, len(self.at))
        for p in range(len(self.at) - 1):
            a, b = self.at[p], self.at[p + 1]
            # _ready removed adjacent edges, so no pending gate joins a and b.
            delta = balance[a] - balance[b]
            if delta < 0:
                key = (delta, 0 if self._fusible(p) else 1, p)
                best = min(best, key)
        return best[2] if best[0] < 0 else None

    def diagonal(self, pending: Dict[int, Instruction], strategy: str) -> None:
        if strategy == "swap_network":
            self._ready(pending)
            pairs: Dict[Pair, List[int]] = {}
            for origin, instruction in pending.items():
                index = instruction["index"]
                pairs.setdefault((min(index), max(index)), []).append(origin)
            for layer in range(len(self.at)):
                if not pending:
                    return
                for p in range(layer % 2, len(self.at) - 1, 2):
                    a, b = self.at[p], self.at[p + 1]
                    for origin in pairs.pop((min(a, b), max(a, b)), []):
                        self.gate(pending.pop(origin), origin)
                    if not pending:
                        return
                    self.swap(p)
            if pending:
                raise RuntimeError("swap network did not execute all pending gates")
            return
        while pending:
            self._ready(pending)
            if not pending:
                return
            swap_position = self._greedy_swap(pending)
            if swap_position is not None:
                self.swap(swap_position)
            else:
                origin = min(
                    pending,
                    key=lambda k: (
                        abs(
                            self.pos[pending[k]["index"][0]]
                            - self.pos[pending[k]["index"][1]]
                        ),
                        k,
                    ),
                )
                self.ordinary(pending.pop(origin), origin)


def linear_route(
    circuit: Circuit,
    info: Optional[Dict[str, Any]] = None,
    initial_layout: Optional[Sequence[int]] = None,
    strategy: str = "auto",
    commute_diagonal: bool = True,
    fuse: bool = True,
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Circuit, Dict[str, Any]]:
    """
    Route a circuit to a line, retaining the final qubit permutation.

    This deterministic host-side pass accepts bound one- and two-qubit gates
    on the default zero input. Channels, extra instructions, MPO gates, and
    higher-arity gates are unsupported. Gate tensors and fusion use the active
    backend; routing decisions never inspect gate values. Gates are assumed
    unitary. Global phase is preserved.

    :param circuit: Input qubit Circuit. The circuit and its QIR are not mutated.
    :param info: Previous pass metadata. Its logical-to-physical mapping is
        composed with this pass's final layout; unrelated fields are preserved.
    :param initial_layout: Logical-to-physical permutation of the input qubits.
        By default choose RCM only if it reduces the interaction bandwidth.
    :param strategy: ``"greedy"``, ``"swap_network"``, or ``"auto"``. Auto
        routes both ways and chooses fewer output two-qubit gates, preferring
        greedy on a tie. Both strategies preserve ordinary gate order.
    :param commute_diagonal: Reorder consecutive known diagonal gates. Ready
        gates sweep along the chain starting near the previous center.
    :param fuse: Merge two-qubit gates on the same physical pair when no
        intervening gate touches either site, including inserted SWAPs.
        Ordinary unmerged gates retain their index orientation, factory,
        parameters, and split settings. Fusion produces dense ``any`` gates.
    :param compiled_options: Pipeline options overriding the named arguments.
    :return: ``(routed, info)``. ``initial_layout`` and ``final_layout`` are
        permutations relative to this pass's input, with ``layout[l]`` the
        physical position of logical qubit ``l``. Decode a physical sample
        ``y`` as ``x[l] = y[final_layout[l]]``. The accumulated
        ``logical_physical_mapping`` instead refers to the pipeline's original
        input. ``origins[k]`` lists input QIR positions contributing to output
        instruction ``k``; inserted SWAPs contribute no origin. ``num_swaps``
        counts inserted SWAPs before fusion. The two-qubit counts describe
        actual input/output instructions. ``bandwidth`` and ``cutwidth`` each
        contain ``identity`` and ``initial`` values on unique interaction pairs.

    For arbitrary input states, the operator identity is
    ``V P_initial = P_final U``. Permutation-invariant zero inputs need only
    the final layout when interpreting the result. Different routed schedules
    can incur different errors under MPS truncation at a fixed bond dimension.

    .. code-block:: python

        routed, info = tc.compiler.linear_route(circuit)
        mps = tc.MPSCircuit.from_qir(
            routed.to_qir(),
            {"nqubits": circuit._nqubits, "split": {"max_singular_values": 64}},
        )
        physical_samples = mps.sample(batch=100, format="sample_bin")
        logical_samples = physical_samples[:, info["final_layout"]]
    """
    options = dict(compiled_options or {})
    unknown = options.keys() - {
        "initial_layout",
        "strategy",
        "commute_diagonal",
        "fuse",
    }
    if unknown:
        raise ValueError(f"unknown routing options: {sorted(unknown)}")
    initial_layout = options.get("initial_layout", initial_layout)
    strategy = options.get("strategy", strategy)
    commute_diagonal = options.get("commute_diagonal", commute_diagonal)
    fuse = options.get("fuse", fuse)
    if strategy not in ("greedy", "swap_network", "auto"):
        raise ValueError("strategy must be greedy, swap_network, or auto")
    qir, edges, diagonal = _prepare(circuit)
    n = circuit._nqubits
    layout = (
        _initial_layout(n, edges)
        if initial_layout is None
        else _layout(initial_layout, n)
    )
    strategies = [strategy]
    if strategy == "auto":
        strategies = ["greedy"]
        if commute_diagonal and diagonal:
            strategies.append("swap_network")
    candidates = []
    for method in strategies:
        router = _Router(layout, fuse)
        i = 0
        while i < len(qir):
            if commute_diagonal and i in diagonal:
                end = i + 1
                while end < len(qir) and end in diagonal:
                    end += 1
                router.diagonal({j: qir[j] for j in range(i, end)}, method)
                i = end
            else:
                router.ordinary(qir[i], i)
                i += 1
        count = sum(len(d["index"]) == 2 for d in router.qir)
        candidates.append((count, method, router))
    count, method, router = min(candidates, key=lambda item: item[0])
    result = dict(info or {})
    old_mapping = result.get("logical_physical_mapping", {q: q for q in range(n)})
    if any(
        not isinstance(p, Integral) or not 0 <= int(p) < n for p in old_mapping.values()
    ):
        raise ValueError(
            "previous mapping contains positions outside the input circuit"
        )
    bw0, cw0 = _widths(edges, list(range(n)))
    bw1, cw1 = _widths(edges, layout)
    result.update(
        initial_layout=list(layout),
        final_layout=list(router.pos),
        logical_physical_mapping={q: router.pos[p] for q, p in old_mapping.items()},
        strategy=method,
        num_swaps=router.num_swaps,
        num_two_qubit_gates=count,
        num_two_qubit_gates_input=sum(len(d["index"]) == 2 for d in qir),
        bandwidth={"identity": bw0, "initial": bw1},
        cutwidth={"identity": cw0, "initial": cw1},
        origins=router.origins,
    )
    return Circuit.from_qir(router.qir, dict(circuit.circuit_param)), result  # type: ignore[return-value]
