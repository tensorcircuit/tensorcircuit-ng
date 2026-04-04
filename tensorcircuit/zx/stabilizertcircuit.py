"""
Stabilizer+T Circuit class using ZX-calculus and JAX.
"""

from __future__ import annotations
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re

import jax
import jax.numpy as jnp
import numpy as np
import pyzx_param as pyzx
from pyzx_param.simulate import DecompositionStrategy
from fractions import Fraction

from ..cons import rdtypestr

from .evaluator import evaluate
from .scalar_graph import (
    compile_program,
    CompiledProgram,
    CompiledComponent,
    find_stab,
    compile_scalar_graphs,
)
from .converter import (
    prepare_graph,
    circuit_to_zx,
    build_sampling_graph,
    build_amplitude_graph,
    transform_error_basis,
    GATE_TABLE,
)
from .noise_model import ChannelSampler
from .utils import get_params
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
            s, key, _ = sample_component(comp, f_params, key)
        else:
            s, key, _ = _sample_component_jit(comp, f_params, key)
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
        self._compiled_probs: Optional[CompiledProgram] = None
        self._channel_sampler: Optional[ChannelSampler] = None
        self._channel_sampler_probs: Optional[ChannelSampler] = None
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
            assert self._channel_sampler is not None
            f_zeros = jnp.zeros(
                (1, self._channel_sampler.num_f_params), dtype=jnp.bool_
            )
            assert self._compiled_program is not None
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
        has_m = any(
            d.get("name", "").upper()
            in ["MEASURE", "M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ"]
            for d in self._qir
        )
        force_measure_all = (not has_m) and (len(state) == self._nqubits)

        if self._compiled_probs is None:
            prepared = prepare_graph(
                self, sample_detectors=False, force_measure_all=force_measure_all
            )
            self._compiled_probs = compile_program(
                prepared, mode="joint", strategy=self.strategy
            )
            self._channel_sampler_probs = ChannelSampler(
                prepared.channel_probs, prepared.error_transform, seed=self._seed
            )

        f_samples = jnp.asarray(self._channel_sampler_probs.sample(shots))
        p_norm = jnp.ones(shots)
        p_joint = jnp.ones(shots)

        for component in self._compiled_probs.components:
            assert len(component.compiled_scalar_graphs) == 2

            f_selected = f_samples[:, component.f_selection]

            norm_circuit, joint_circuit = component.compiled_scalar_graphs

            # Normalization: only f-params
            p_norm = p_norm * jnp.abs(evaluate(norm_circuit, f_selected))

            # Joint probability: f-params + state
            component_state = state[jnp.array(component.output_indices)]
            tiled_state = jnp.tile(component_state, (shots, 1))
            joint_params = jnp.hstack([f_selected, tiled_state])
            p_joint = p_joint * jnp.abs(evaluate(joint_circuit, joint_params))

        return p_joint / p_norm

    def amplitude(self, state: Union[jax.Array, Sequence[int], str]) -> jax.Array:
        """
        Calculate the complex amplitude <state|psi> for a noiseless circuit.
        Fails if the circuit contains non-unitary operations or noise.

        :param state: The target bitstring.
        :type state: Union[jax.Array, Sequence[int], str]
        :return: The complex amplitude.
        :rtype: jax.Array
        """
        if isinstance(state, str):
            state = [int(x) for x in state]
        built = circuit_to_zx(self, force_measure_all=True)
        if built.num_error_bits > 0:
            raise ValueError("amplitude() only supported for noiseless circuits.")

        graph = build_amplitude_graph(built, state)
        pyzx.full_reduce(graph, paramSafe=True)
        graphs = find_stab(graph, strategy=self.strategy)
        compiled = compile_scalar_graphs(graphs, [])
        dummy_f = jnp.zeros((1, 0), dtype=jnp.bool_)
        amp = evaluate(compiled, dummy_f)
        # Divide by sqrt(2)^n due to ZX boundary conventions
        return amp[0] / (2.0 ** (self._nqubits / 2.0))

    def expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        ps: Optional[Sequence[int]] = None,
        nmc: int = 1000,
        **kwargs: Any,
    ) -> jax.Array:
        """
        Calculate the expectation value <psi|O|psi> for a Pauli string O.
        Supports noiseless and noisy circuits.

        :param x: Qubit indices for X operators.
        :type x: Optional[Sequence[int]]
        :param y: Qubit indices for Y operators.
        :type y: Optional[Sequence[int]]
        :param z: Qubit indices for Z operators.
        :type z: Optional[Sequence[int]]
        :param ps: Pauli string as a sequence of integers (0:I, 1:X, 2:Y, 3:Z).
        :type ps: Optional[Sequence[int]]
        :param nmc: Number of Monte Carlo trajectories for noisy circuits.
        :type nmc: int
        :return: The expectation value.
        :rtype: jax.Array
        """
        pauli_dict: Dict[int, str] = {}
        if ps is not None:
            for i, p in enumerate(ps):
                if p == 1:
                    pauli_dict[i] = "X"
                elif p == 2:
                    pauli_dict[i] = "Y"
                elif p == 3:
                    pauli_dict[i] = "Z"
        if x is not None:
            for i in x:
                pauli_dict[i] = "X"
        if y is not None:
            for i in y:
                pauli_dict[i] = "Y"
        if z is not None:
            for i in z:
                pauli_dict[i] = "Z"

        prepared = prepare_graph(
            self,
            sample_detectors=False,
            force_measure_all=False,
            pauli=pauli_dict,
            reset_scalar=False,
        )
        graphs = find_stab(prepared.graph, strategy=self.strategy)
        param_names = sorted(
            [
                p
                for p in get_params(prepared.graph)
                if p != "1" and (p.startswith("e") or p.startswith("f"))
            ],
            key=lambda p: (p[0], int(p[1:])),
        )
        compiled = compile_scalar_graphs(graphs, param_names)

        if prepared.num_error_bits == 0:
            vals = evaluate(compiled, jnp.zeros((1, 0), dtype=jnp.bool_))
            # Divide by 2^n due to ZX doubled boundary conventions
            return vals[0] / (2.0**self._nqubits)

        if nmc <= 0:
            raise ValueError("nmc must be positive for noisy expectation_ps().")

        self._key, subkey = jax.random.split(self._key)
        mc_seed = int(
            np.asarray(
                jax.random.randint(
                    subkey, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
                )
            )
        )
        rng = np.random.default_rng(mc_seed)
        sampled_e = np.zeros((nmc, prepared.num_error_bits), dtype=np.uint8)
        e_offset = 0
        for probs in prepared.channel_probs:
            probs = np.asarray(probs, dtype=np.float64)
            nbits = int(np.log2(len(probs)))
            outcomes = rng.choice(len(probs), size=nmc, p=probs)
            bits = ((outcomes[:, None] >> np.arange(nbits)) & 1).astype(np.uint8)
            sampled_e[:, e_offset : e_offset + nbits] = bits
            e_offset += nbits

        sampled_f = (
            (sampled_e.astype(np.uint8) @ prepared.error_transform.T.astype(np.uint8))
            % 2
            if prepared.error_transform.size > 0
            else np.zeros((nmc, 0), dtype=np.uint8)
        )
        param_cols: list[np.ndarray] = []
        for p in param_names:
            idx = int(p[1:])
            if p.startswith("e"):
                param_cols.append(sampled_e[:, idx])
            else:
                param_cols.append(sampled_f[:, idx])

        if param_cols:
            param_values = jnp.asarray(np.stack(param_cols, axis=1), dtype=jnp.bool_)
        else:
            param_values = jnp.zeros((nmc, 0), dtype=jnp.bool_)

        vals = evaluate(compiled, param_values)
        return jnp.mean(vals) / (2.0**self._nqubits)

    def _sample_batches(self, shots: int, batch_size: int = 1000) -> jax.Array:
        batches = []
        for _ in range(ceil(shots / batch_size)):
            assert self._channel_sampler is not None
            f_params = jnp.asarray(self._channel_sampler.sample(batch_size))
            self._key, subkey = jax.random.split(self._key)
            assert self._compiled_program is not None
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
        if name.upper() in GATE_TABLE:

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

    def detector_instruction(  # type: ignore[override]
        self,
        lookback_indices: list[int],
        coords: Optional[list[float]] = None,
    ) -> None:
        self._qir.append(
            {"name": "DETECTOR", "index": lookback_indices, "coords": coords}
        )

    def observable_instruction(  # type: ignore[override]
        self, lookback_indices: list[int], observable_index: int = 0
    ) -> None:
        self._qir.append(
            {
                "name": "OBSERVABLE_INCLUDE",
                "index": lookback_indices,
                "observable_index": observable_index,
            }
        )

    def qubit_coords_instruction(self, qubit: int, coords: list[float]) -> None:
        self._qir.append({"name": "QUBIT_COORDS", "index": [qubit], "coords": coords})

    def tick_instruction(self) -> None:
        self._qir.append({"name": "TICK"})

    def reset_instruction(self, q: int) -> None:  # type: ignore[override]
        self._qir.append({"name": "RESET", "index": [q]})

    def measure_instruction(self, q: int, p: float = 0) -> None:  # type: ignore[override]
        self._qir.append({"name": "MEASURE", "index": [q], "p": p})

    def mr_instruction(self, q: int, p: float = 0) -> None:  # type: ignore[override]
        self._qir.append({"name": "MR", "index": [q], "p": p})

    def mrx_instruction(self, q: int, p: float = 0) -> None:
        self._qir.append({"name": "MRX", "index": [q], "p": p})

    def mry_instruction(self, q: int, p: float = 0) -> None:
        self._qir.append({"name": "MRY", "index": [q], "p": p})

    def mrz_instruction(self, q: int, p: float = 0) -> None:
        self._qir.append({"name": "MRZ", "index": [q], "p": p})

    def rx(self, q: int, theta: float = 0) -> None:
        self._qir.append({"name": "R_X", "index": [q], "parameters": {"theta": theta}})

    def ry(self, q: int, theta: float = 0) -> None:
        self._qir.append({"name": "R_Y", "index": [q], "parameters": {"theta": theta}})

    def rz(self, q: int, theta: float = 0) -> None:
        self._qir.append({"name": "R_Z", "index": [q], "parameters": {"theta": theta}})

    def depolarizing(self, q: int, p: float) -> None:
        self._qir.append({"name": "DEPOLARIZE1", "index": [q], "parameters": {"p": p}})

    def depolarizing2(self, q1: int, q2: int, p: float) -> None:
        self._qir.append(
            {"name": "DEPOLARIZE2", "index": [q1, q2], "parameters": {"p": p}}
        )

    def depolarizing_instruction(self, q: int, p: float) -> None:  # type: ignore[override]
        self.depolarizing(q, p)

    def depolarizing2_instruction(self, q1: int, q2: int, p: float) -> None:  # type: ignore[override]
        self.depolarizing2(q1, q2, p)

    def pauli_instruction(  # type: ignore[override]
        self, q: int, px: float = 0, py: float = 0, pz: float = 0
    ) -> None:
        self._qir.append(
            {
                "name": "PAULI_CHANNEL_1",
                "index": [q],
                "parameters": {"px": px, "py": py, "pz": pz},
            }
        )

    def pauli(self, q: int, probs: list[float]) -> None:
        self._qir.append(
            {"name": "PAULI_CHANNEL_1", "index": [q], "parameters": {"probs": probs}}
        )

    def x_error(self, q: int, p: float) -> None:
        self._qir.append({"name": "X_ERROR", "index": [q], "parameters": {"p": p}})

    def y_error(self, q: int, p: float) -> None:
        self._qir.append({"name": "Y_ERROR", "index": [q], "parameters": {"p": p}})

    def z_error(self, q: int, p: float) -> None:
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

            if name == "I" and instruction.tag:
                match = re.match(r"^(\w+)\((.*)\)$", instruction.tag)
                if match:
                    gate_name, params_str = match.groups()
                    params = {}
                    for param in params_str.split(","):
                        if "=" in param:
                            k, v = param.strip().split("=")
                            params[k] = (
                                float(v[:-3]) * np.pi if v.endswith("*pi") else float(v)
                            )
                    inst._qir.append(
                        {"name": gate_name, "index": targets, "parameters": params}
                    )
                    continue

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
