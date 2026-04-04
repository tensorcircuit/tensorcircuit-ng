"""
Stabilizer+T Circuit class using ZX-calculus and JAX.
"""

from __future__ import annotations
from math import ceil
from typing import Any, Dict, List, Optional, cast, Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from pyzx_param.simulate import DecompositionStrategy

from ..cons import rdtypestr

from .evaluator import evaluate
from .scalar_graph import compile_program, CompiledProgram, CompiledComponent
from .converter import prepare_graph, circuit_to_zx, build_sampling_graph
from .noise_model import ChannelSampler
from ..abstractcircuit import AbstractCircuit


def sample_component(
    comp: CompiledComponent, f_params: jax.Array, key: Any
) -> Tuple[jax.Array, Any, jax.Array]:
    """
    Sample measurement outcomes for a single compiled component.

    :param comp: The compiled component to sample from.
    :type comp: CompiledComponent
    :param f_params: The sampled error bit parameters (f-basis).
    :type f_params: jax.Array
    :param key: JAX PRNG key for sampling.
    :type key: Any
    :return: A tuple containing the measurement samples, next PRNG key, and maximum deviation.
    :rtype: Tuple[jax.Array, Any, jax.Array]
    """
    batch_size = f_params.shape[0]
    num_outputs = len(comp.compiled_scalar_graphs) - 1
    f_selected = f_params[:, comp.f_selection].astype(jnp.bool_)
    m_acc = jnp.zeros((batch_size, num_outputs), dtype=jnp.bool_)
    prev = jnp.abs(evaluate(comp.compiled_scalar_graphs[0], f_selected))
    ones, zero = jnp.ones((batch_size, 1), dtype=jnp.bool_), jnp.zeros(
        (1, 1), dtype=jnp.bool_
    )
    max_dev = jnp.array(0.0, dtype=rdtypestr)
    for i, circuit in enumerate(comp.compiled_scalar_graphs[1:]):
        params = jnp.hstack([f_selected, m_acc[:, :i], ones])
        check_row = jnp.hstack([f_selected[:1], m_acc[:1, :i], zero])
        probs = jnp.abs(evaluate(circuit, jnp.vstack([params, check_row])))
        p1, p0_single = probs[:batch_size], probs[-1]
        norm = (p0_single + p1[0]) / prev[0]
        max_dev = jnp.maximum(max_dev, jnp.abs(norm - 1.0))
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)
        m_acc = m_acc.at[:, i].set(bits)
        prev = jnp.where(bits, p1, prev - p1)
    return m_acc, key, max_dev


@jax.jit
def _sample_component_jit(
    comp: CompiledComponent, f_params: jax.Array, key: Any
) -> tuple[jax.Array, Any, jax.Array]:
    return sample_component(comp, f_params, key)


def sample_program(
    program: CompiledProgram, f_params: jax.Array, key: Any
) -> jax.Array:
    """
    Sample measurement outcomes for an entire compiled program.

    :param program: The compiled program to sample from.
    :type program: CompiledProgram
    :param f_params: The sampled error bit parameters (f-basis).
    :type f_params: jax.Array
    :param key: JAX PRNG key for sampling.
    :type key: Any
    :return: The measurement samples for all qubits.
    :rtype: jax.Array
    """
    results = []
    for comp in program.components:
        if len(comp.output_indices) <= 1:
            s, key, dev = sample_component(comp, f_params, key)
        else:
            s, key, dev = _sample_component_jit(comp, f_params, key)
        results.append(s)
    if not results:
        return jnp.zeros(
            (f_params.shape[0], len(program.output_order)), dtype=jnp.bool_
        )
    combined = jnp.concatenate(results, axis=1)
    return combined[:, jnp.argsort(jnp.array(program.output_order))]


class StabilizerTCircuit(AbstractCircuit):
    def __init__(
        self,
        nqubits: int,
        seed: Optional[int] = None,
        strategy: DecompositionStrategy = "cat5",
    ):
        """
        Initialize a StabilizerTCircuit.

        :param nqubits: Number of qubits in the circuit.
        :type nqubits: int
        :param seed: Random seed for sampling, defaults to None.
        :type seed: Optional[int], optional
        :param strategy: Decomposition strategy for T gates, defaults to "cat5".
        :type strategy: DecompositionStrategy, optional
        """
        self._nqubits = nqubits
        self._qir = []
        self._extra_qir = []
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))
        self._seed = seed
        self._key = jax.random.key(seed)
        self.strategy = strategy
        self._compiled_program: Optional[CompiledProgram] = None
        self._channel_sampler: Optional[ChannelSampler] = None
        self._num_detectors = 0
        self._num_observables = 0

    def _merge_qir(self) -> List[Dict[str, Any]]:
        """
        Merge _qir and _extra_qir into a single list of instructions ordered by 'pos'.
        """
        return self._qir

    def _compile(self, sample_detectors: bool, force_measure_all: bool = False) -> None:
        prepared = prepare_graph(
            self, sample_detectors=sample_detectors, force_measure_all=force_measure_all
        )
        self._compiled_program = compile_program(
            prepared, mode="sequential", strategy=self.strategy
        )
        self._channel_sampler = ChannelSampler(
            prepared.channel_probs, prepared.error_transform, seed=self._seed
        )
        self._num_detectors = prepared.num_detectors
        self._num_observables = len(prepared.observables)

    @classmethod
    def from_circuit(
        cls, circuit: AbstractCircuit, strategy: DecompositionStrategy = "cat5"
    ) -> StabilizerTCircuit:
        """
        Create a StabilizerTCircuit from an existing TensorCircuit AbstractCircuit.

        :param circuit: The source circuit to convert.
        :type circuit: AbstractCircuit
        :param strategy: Decomposition strategy for T gates, defaults to "cat5".
        :type strategy: DecompositionStrategy, optional
        :return: A new StabilizerTCircuit instance.
        :rtype: StabilizerTCircuit
        """
        stc = cls(circuit._nqubits, strategy=strategy)

        # Collect all operations with their position
        all_ops = []
        for i, d in enumerate(circuit._qir):
            new_d = d.copy()
            if "gatef" in new_d:
                if "name" not in new_d or not new_d["name"]:
                    gatef = new_d["gatef"]
                    if hasattr(gatef, "name"):
                        new_d["name"] = gatef.name.upper()
                    elif hasattr(gatef, "__name__"):
                        new_d["name"] = gatef.__name__.upper()
            if "name" in new_d:
                new_d["name"] = new_d["name"].upper()
            all_ops.append((float(i), new_d))

        for d in getattr(circuit, "_extra_qir", []):
            new_d = d.copy()
            if "name" in new_d:
                new_d["name"] = new_d["name"].upper()
            # Interleave based on 'pos' field
            pos = float(new_d.get("pos", len(circuit._qir)))
            # Ties: instructions at same 'pos' as a gate should typically come after the gate?
            # Or before? AbstractCircuit appends to _extra_qir AFTER the gate is added to _qir.
            # So pos = len(_qir) before the gate? No, l = len(_qir) is the index of the gate about to be added.
            # So it should be pos + some small epsilon to be consistent with the order they were called.
            all_ops.append((pos + 0.1, new_d))

        # Sort by position
        all_ops.sort(key=lambda x: x[0])
        stc._qir = [op for _, op in all_ops]

        return stc

    def sample_measurements(
        self, shots: int = 1, seed: Optional[int] = None, batch_size: int = 1000
    ) -> jax.Array:
        """
        Sample all measurement outcomes in the circuit.

        :param shots: Number of samples to draw, defaults to 1.
        :type shots: int, optional
        :param seed: Random seed for this sampling run, defaults to None.
        :type seed: Optional[int], optional
        :param batch_size: Number of shots per JIT batch, defaults to 1000.
        :type batch_size: int, optional
        :return: Array of measurement samples with shape (shots, num_measurements).
        :rtype: jax.Array
        """
        if seed is not None:
            self._key = jax.random.key(seed)
        has_m = any(
            d.get("name", "").upper()
            in ["MEASURE", "M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ"]
            for d in self._qir
        )
        if (
            self._compiled_program is None
            or self._compiled_program.mode != "sequential"
        ):
            self._compile(sample_detectors=False, force_measure_all=not has_m)
        return self._sample_batches(shots, batch_size)

    def sample_detectors(
        self,
        shots: int = 1,
        separate_observables: bool = False,
        use_reference: bool = False,
        seed: Optional[int] = None,
        batch_size: int = 1000,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """
        Sample detector and observable outcomes.

        :param shots: Number of samples to draw, defaults to 1.
        :type shots: int, optional
        :param separate_objects: Whether to return detectors and observables separately, defaults to False.
        :type separate_objects: bool, optional
        :param use_reference: Whether to XOR results with a noiseless reference run, defaults to False.
        :type use_reference: bool, optional
        :param seed: Random seed for this sampling run, defaults to None.
        :type seed: Optional[int], optional
        :param batch_size: Number of shots per JIT batch, defaults to 1000.
        :type batch_size: int, optional
        :return: Array of samples or tuple of (detectors, observables) arrays.
        :rtype: Union[jax.Array, Tuple[jax.Array, jax.Array]]
        """
        if seed is not None:
            self._key = jax.random.key(seed)
        if (
            self._compiled_program is None
            or self._compiled_program.mode != "sequential"
        ):
            self._compile(sample_detectors=True)

        samples = self._sample_batches(shots, batch_size)

        if use_reference:
            # For reference, we need f_params = 0
            f_zeros = jnp.zeros(
                (1, self._channel_sampler.num_f_params), dtype=jnp.bool_
            )
            ref_sample = sample_program(self._compiled_program, f_zeros, self._key)
            samples = samples ^ ref_sample

        if separate_observables:
            detectors = samples[:, : self._num_detectors]
            observables = samples[
                :, self._num_detectors : self._num_detectors + self._num_observables
            ]
            return detectors, observables

        return samples[:, : self._num_detectors + self._num_observables]

    def outcome_probability(self, state: jax.Array, shots: int = 1) -> jax.Array:
        """
        Compute the probability of a specific measurement outcome state.

        :param state: The target measurement bitstring.
        :type state: jax.Array
        :param shots: Number of noise realizations to average over, defaults to 1.
        :type shots: int, optional
        :return: Probability of the outcome state for each noise realization.
        :rtype: jax.Array
        """
        # 1. Prepare graphs
        has_m = any(
            d.get("name", "").upper()
            in ["MEASURE", "M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ"]
            for d in self._qir
        )
        force_measure_all = (not has_m) and (len(state) == self._nqubits)

        # Re-implementing based on tsim.CompiledStateProbs:
        # We use sample_detectors=False to ensure the outputs are the measurement records
        built = circuit_to_zx(self, force_measure_all=force_measure_all)
        prepared = build_sampling_graph(built, sample_detectors=False)
        # We don't reduce here to ensure outputs stay

        g_state = prepared.copy()
        g_state.apply_effect("".join(["1" if b else "0" for b in state]))

        g_norm = prepared.copy()
        g_norm.apply_effect("+" * len(state))

        # Compensate power for trace of unplugged outputs
        # In joint mode, we plug all outputs of the component.
        # But here we are plugging all outputs of the WHOLE graph.
        # tsim: g.scalar.add_power(num_outputs - num_plugged)
        # For g_state: num_plugged = len(state), so add_power(0)
        # For g_norm: num_plugged = 0 (since it's all '+'), so add_power(len(state))
        g_norm.scalar.add_power(len(state))

        # Compile them
        # find_stab expects GraphS
        from .scalar_graph import find_stab, compile_scalar_graphs

        stabs_state = find_stab(g_state, strategy=self.strategy)
        stabs_norm = find_stab(g_norm, strategy=self.strategy)

        # Transform error basis to get f-params
        from .converter import transform_error_basis

        prepared, error_transform = transform_error_basis(
            prepared, num_e=built.num_error_bits
        )

        from .utils import get_params

        active = get_params(prepared)  # Use transformed graph's vars
        f_vars = sorted([v for v in active if v.startswith("f")])

        compiled_state = compile_scalar_graphs(stabs_state, f_vars)
        compiled_norm = compile_scalar_graphs(stabs_norm, f_vars)

        # Sample f-params
        channel_sampler = ChannelSampler(
            built.channel_probs, error_transform, seed=self._seed
        )
        f_samples = jnp.asarray(channel_sampler.sample(shots))
        # f_samples has all f-params, but evaluate expects only the ones in f_vars
        # Actually, f_vars are the names like 'f0', 'f1'.
        # We need to map these to indices.
        f_indices = [int(v[1:]) for v in f_vars]
        f_selected = (
            f_samples[:, jnp.array(f_indices)]
            if f_indices
            else jnp.zeros((shots, 0), dtype=jnp.bool_)
        )

        p_state = jnp.abs(evaluate(compiled_state, f_selected))
        p_norm = jnp.abs(evaluate(compiled_norm, f_selected))

        return p_state / p_norm

    def _sample_batches(self, shots: int, batch_size: int = 1000) -> jax.Array:
        batches = []
        for _ in range(ceil(shots / batch_size)):
            f_params = jnp.asarray(self._channel_sampler.sample(batch_size))
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(self._compiled_program, f_params, subkey)
            batches.append(samples)
        return jnp.concatenate(batches, axis=0)[:shots]

    def apply(self, gate: Any, *index: int, **kwargs: Any) -> None:
        if hasattr(gate, "name"):
            name = gate.name.upper()
        else:
            name = ""
        self._qir.append({"name": name, "index": list(index), "parameters": kwargs})

    def apply_general_gate(
        self,
        gate: Any,
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        diagonal: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ir_dict:
            self._qir.append(ir_dict)
        else:
            if name is None and hasattr(gate, "name"):
                name = gate.name
            self._qir.append(
                {"name": name.upper() if name else "", "index": list(index)}
            )

    def __getattr__(self, name: str) -> Any:
        if name in self.defined_gates:

            def wrapper(*index: int, **kwargs: Any) -> None:
                self._qir.append(
                    {"name": name.upper(), "index": list(index), "parameters": kwargs}
                )

            return wrapper
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # Simplified StabilizerTCircuit methods for compatibility
    def h(self, q: int) -> None:
        self._qir.append({"name": "H", "index": [q]})

    def cnot(self, c: int, t: int) -> None:
        self._qir.append({"name": "CNOT", "index": [c, t]})

    def cx(self, c: int, t: int) -> None:
        self.cnot(c, t)

    def cz(self, c: int, t: int) -> None:
        self._qir.append({"name": "CZ", "index": [c, t]})

    def x(self, q: int) -> None:
        self._qir.append({"name": "X", "index": [q]})

    def y(self, q: int) -> None:
        self._qir.append({"name": "Y", "index": [q]})

    def z(self, q: int) -> None:
        self._qir.append({"name": "Z", "index": [q]})

    def s(self, q: int) -> None:
        self._qir.append({"name": "S", "index": [q]})

    def sd(self, q: int) -> None:
        self._qir.append({"name": "SD", "index": [q]})

    def t(self, q: int) -> None:
        self._qir.append({"name": "T", "index": [q]})

    def td(self, q: int) -> None:
        self._qir.append({"name": "TD", "index": [q]})

    def swap(self, q1: int, q2: int) -> None:
        self._qir.append({"name": "SWAP", "index": [q1, q2]})

    def detector_instruction(
        self, lookback_indices: list[int], coords: list[float] = None
    ):
        self._qir.append(
            {"name": "DETECTOR", "index": lookback_indices, "coords": coords}
        )

    def observable_instruction(
        self, lookback_indices: list[int], observable_index: int = 0
    ):
        self._qir.append(
            {
                "name": "OBSERVABLE_INCLUDE",
                "index": lookback_indices,
                "observable_index": observable_index,
            }
        )

    def qubit_coords_instruction(self, qubit: int, coords: list[float]):
        self._qir.append({"name": "QUBIT_COORDS", "index": [qubit], "coords": coords})

    def tick_instruction(self):
        self._qir.append({"name": "TICK"})

    def reset_instruction(self, q: int):
        self._qir.append({"name": "RESET", "index": [q]})

    def measure_instruction(self, q: int, p: float = 0):
        self._qir.append({"name": "MEASURE", "index": [q], "p": p})

    def mr_instruction(self, q: int, p: float = 0):
        self._qir.append({"name": "MR", "index": [q], "p": p})

    def mrx_instruction(self, q: int, p: float = 0):
        self._qir.append({"name": "MRX", "index": [q], "p": p})

    def mry_instruction(self, q: int, p: float = 0):
        self._qir.append({"name": "MRY", "index": [q], "p": p})

    def mrz_instruction(self, q: int, p: float = 0):
        self._qir.append({"name": "MRZ", "index": [q], "p": p})

    def depolarizing(self, q: int, p: float):
        self._qir.append({"name": "DEPOLARIZE1", "index": [q], "parameters": {"p": p}})

    def depolarizing2(self, q1: int, q2: int, p: float):
        self._qir.append(
            {"name": "DEPOLARIZE2", "index": [q1, q2], "parameters": {"p": p}}
        )

    def depolarizing_instruction(self, q: int, p: float):
        self.depolarizing(q, p)

    def depolarizing2_instruction(self, q1: int, q2: int, p: float):
        self.depolarizing2(q1, q2, p)

    def pauli_instruction(self, q: int, px: float = 0, py: float = 0, pz: float = 0):
        self._qir.append(
            {
                "name": "PAULI_CHANNEL_1",
                "index": [q],
                "parameters": {"px": px, "py": py, "pz": pz},
            }
        )

    def pauli(self, q: int, probs: list[float]):
        self._qir.append(
            {"name": "PAULI_CHANNEL_1", "index": [q], "parameters": {"probs": probs}}
        )

    def x_error(self, q: int, p: float):
        self._qir.append({"name": "X_ERROR", "index": [q], "parameters": {"p": p}})

    def y_error(self, q: int, p: float):
        self._qir.append({"name": "Y_ERROR", "index": [q], "parameters": {"p": p}})

    def z_error(self, q: int, p: float):
        self._qir.append({"name": "Z_ERROR", "index": [q], "parameters": {"p": p}})

    @classmethod
    def from_stim_circuit(cls, stim_circuit: Any) -> "StabilizerTCircuit":
        """
        Create a StabilizerTCircuit from a stim.Circuit object.

        :param stim_circuit: The stim circuit to convert.
        :type stim_circuit: Any
        :return: A new StabilizerTCircuit instance.
        :rtype: StabilizerTCircuit
        """
        inst = cls(stim_circuit.num_qubits)
        # We directly populate inst._qir for efficiency
        for instruction in stim_circuit.flattened():
            name = instruction.name
            if name in ["QUBIT_COORDS", "SHIFT_COORDS", "I_ERROR"]:
                continue

            targets = [t.value for t in instruction.targets_copy()]
            args = instruction.gate_args_copy()

            if name == "TICK":
                inst._qir.append({"name": "TICK"})
                continue

            if name == "DETECTOR":
                inst._qir.append({"name": "DETECTOR", "index": targets})
                continue

            if name == "OBSERVABLE_INCLUDE":
                inst._qir.append(
                    {"name": "OBSERVABLE_INCLUDE", "index": targets, "p": int(args[0])}
                )
                continue

            # Map stim names to TC _qir names
            # Reference: tsim.core.instructions.GATE_TABLE
            num_qubits = 1
            if name in [
                "CNOT",
                "CX",
                "CZ",
                "CY",
                "SWAP",
                "DEPOLARIZE2",
                "PAULI_CHANNEL_2",
            ]:
                num_qubits = 2
            elif name in ["SQRT_XX", "SQRT_YY", "SQRT_ZZ", "ISWAP"]:
                num_qubits = 2
            elif name.startswith(("XC", "YC", "ZC")):
                num_qubits = 2

            for i in range(0, len(targets), num_qubits):
                chunk = targets[i : i + num_qubits]
                qir_item: Dict[str, Any] = {"name": name, "index": chunk}

                # Special parameter handling for noise and measurements
                if name in ["DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"]:
                    qir_item["parameters"] = {"p": args[0]}
                elif name == "DEPOLARIZE2":
                    qir_item["parameters"] = {"p": args[0]}
                elif name == "PAULI_CHANNEL_1":
                    qir_item["parameters"] = {
                        "px": args[0],
                        "py": args[1],
                        "pz": args[2],
                    }
                elif name == "M" or name == "MEASURE" or name == "MZ":
                    qir_item["name"] = "MEASURE"
                    if args:
                        qir_item["p"] = args[0]
                elif name in ["MR", "MRX", "MRY", "MRZ"]:
                    if args:
                        qir_item["p"] = args[0]
                elif name in ["MX", "MY"]:
                    if args:
                        qir_item["p"] = args[0]

                inst._qir.append(qir_item)

        return inst

    @classmethod
    def from_stim_str(cls, stim_str: str) -> "StabilizerTCircuit":
        """
        Create a StabilizerTCircuit from a stim circuit string.

        :param stim_str: The stim circuit string to parse.
        :type stim_str: str
        :return: A new StabilizerTCircuit instance.
        :rtype: StabilizerTCircuit
        """
        import stim

        return cls.from_stim_circuit(stim.Circuit(stim_str))


# StabilizerTCircuit._meta_apply()
