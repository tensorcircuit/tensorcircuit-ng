"""
Analog-Digital Hybrid Circuit class wrapper
only support jax backend
"""

from typing import Any, List, Optional, Callable, Dict, Tuple, Union, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import tensornetwork as tn

from .cons import backend, rdtypestr
from .abstractcircuit import defined_gates
from .circuit import Circuit
from .quantum import QuOperator
from .timeevol import ode_evol_global, ode_evol_local
from .utils import arg_alias

Tensor = Any


@dataclass
class AnalogBlock:
    """
    A data structure to hold information about an analog evolution block.
    """

    hamiltonian_func: Callable[[Tensor], Tensor]
    time: float
    index: Optional[List[int]] = None
    solver_options: Optional[Dict[str, Any]] = None


class AnalogCircuit:
    """
    A class for hybrid digital-analog quantum simulation with time-dependent Hamiltonians.
    """

    def __init__(
        self,
        nqubits: int,
        inputs: Optional[Tensor] = None,
        mps_inputs: Optional[QuOperator] = None,
        split: Optional[Dict[str, Any]] = None,
        dim: Optional[int] = None,
    ):
        """
        Initializes the hybrid circuit.

        :param nqubits: The number of qubits in the circuit.
        :type nqubits: int
        :param dim: The local Hilbert space dimension per site. Qudit is supported for 2 <= d <= 36.
        :type dim: If None, the dimension of the circuit will be `2`, which is a qubit system.
        :param inputs: If not None, the initial state of the circuit is taken as ``inputs``
            instead of :math:`\vert 0 \rangle^n` qubits, defaults to None.
        :type inputs: Optional[Tensor], optional
        :param mps_inputs: QuVector for a MPS like initial wavefunction.
        :type mps_inputs: Optional[QuOperator]
        :param split: dict if two qubit gate is ready for split, including parameters for at least one of
            ``max_singular_values`` and ``max_truncation_err``.
        :type split: Optional[Dict[str, Any]]
        """
        self.num_qubits, self._nqubits = nqubits, nqubits
        self.dim = 2**self.num_qubits
        self.inputs = inputs
        if inputs is None:
            self.inputs = np.zeros([self.dim])
            self.inputs[0] = 1.0
            self.inputs = backend.convert_to_tensor(self.inputs)

        # List of digital circuits, starting with one empty circuit.
        self.digital_circuits: List[Circuit] = [
            Circuit(self.num_qubits, inputs, mps_inputs, split, dim)
        ]

        # List of analog blocks, each containing the Hamiltonian function, time, and solver options.
        self.analog_blocks: List[AnalogBlock] = []
        self._effective_circuit: Optional[Circuit] = None
        self._solver_options: Dict[str, Any] = {}

    def set_solver_options(self, **kws: Any) -> None:
        """
        set solver options globally for this circuit object
        """
        self._solver_options = kws

    @property
    def effective_circuit(self) -> Circuit:
        """
        Returns the effective circuit after all blocks have been added.
        """
        if self._effective_circuit is None:
            self.state()
        return self._effective_circuit  # type: ignore

    @property
    def current_digital_circuit(self) -> Circuit:
        """
        Returns the last (currently active) digital circuit.
        """
        return self.digital_circuits[-1]

    def add_analog_block(
        self,
        hamiltonian: Callable[[float], Tensor],
        time: Union[float, List[Tensor]],
        index: Optional[List[int]] = None,
        **solver_options: Any,
    ) -> "AnalogCircuit":
        """
        Adds a time-dependent analog evolution block to the circuit.

        This finalizes the current digital block and prepares a new one for subsequent gates.

        :param hamiltonian_func: A function H(t) that takes a time `t` (from 0 to `time`)
                                 and returns the Hamiltonian matrix at that instant.
        :type hamiltonian_func: Callable[[float], np.ndarray]
        :param time: The total evolution time 'T'.
        :type time: float
        :param index: The indices of the qubits to apply the analog evolution to. Defaults None for
            global application.
        :type index: Optional[List[int]]
        :param solver_options: Keyword arguments passed directly to `tc.timeevol.ode_evolve`
        :type solver_options: Dict[str, Any]
        """
        # Create and store the analog block information
        time = backend.convert_to_tensor(time, dtype=rdtypestr)
        time = backend.reshape(time, [-1])
        if backend.shape_tuple(time)[0] == 1:
            time = backend.stack([0.0, time[0]])  # type: ignore
        elif backend.shape_tuple(time)[0] > 2:
            raise ValueError(
                "Time must be a scalar or a two elements array for the starting and end points."
            )
        combined_solver_options = self._solver_options.copy()
        combined_solver_options.update(solver_options)
        block = AnalogBlock(
            hamiltonian_func=hamiltonian,
            time=time,  # type: ignore
            index=index,
            solver_options=combined_solver_options,
        )
        self.analog_blocks.append(block)

        # After adding an analog block, we start a new digital block.
        self.digital_circuits.append(Circuit(self.num_qubits, inputs=self.inputs))
        self._effective_circuit = None
        return self  # Allow for chaining

    def __getattr__(self, name: str) -> Any:
        """
        Metaprogramming to forward gate calls to the current digital circuit.
        This enables syntax like `analog_circuit.h(0)`.
        """
        gate_method = getattr(self.current_digital_circuit, name, None)

        if gate_method and callable(gate_method) and name.lower() in defined_gates:

            def wrapper(*args, **kwargs):  # type: ignore
                gate_method(*args, **kwargs)
                self._effective_circuit = None
                return self

            return wrapper
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object or its underlying '{type(self.current_digital_circuit).__name__}' "
                f"object has no attribute '{name}'."
            )

    def state(self) -> Tensor:
        """
        Executes the full digital-analog sequence.

        :return: The final state vector after the full evolution
        :rtype: Tensor
        """
        # Propagate the state through the alternating circuit blocks
        for i, analog_block in enumerate(self.analog_blocks):
            # 1. Apply Digital Block i
            digital_c = self.digital_circuits[i]
            if i > 0:
                digital_c.replace_inputs(psi)  # type: ignore
            psi = digital_c.wavefunction()

            if analog_block.index is None:
                psi = ode_evol_global(  # type: ignore
                    hamiltonian=analog_block.hamiltonian_func,
                    initial_state=psi,
                    times=analog_block.time,
                    **analog_block.solver_options,
                )
            else:
                psi = ode_evol_local(  # type: ignore
                    hamiltonian=analog_block.hamiltonian_func,
                    initial_state=psi,
                    times=analog_block.time,
                    index=analog_block.index,
                    **analog_block.solver_options,
                )
            psi = psi[-1]
            # TODO(@refraction-ray): support more time evol methods

        # 3. Apply the final digital circuit
        if self.analog_blocks:
            self.digital_circuits[-1].replace_inputs(psi)
            psi = self.digital_circuits[-1].wavefunction()
        else:
            psi = self.digital_circuits[-1].wavefunction()
        self._effective_circuit = Circuit(self.num_qubits, inputs=psi)

        return psi

    wavefunction = state

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
        :return: Tensor with one element
        :rtype: Tensor
        """
        return self.effective_circuit.expectation(
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
        return self.effective_circuit.measure_jit(
            *index, with_prob=with_prob, status=status
        )

    measure = measure_jit

    def amplitude(self, l: Union[str, Tensor]) -> Tensor:
        r"""
        Return the amplitude for a given bitstring `l`.

        For state simulators, this computes :math:`\langle l \vert \psi \rangle`.

        :param l: Bitstring in base-`d` using `0-9A-Z`.
        :type l: Union[str, Tensor]
        :return: Complex amplitude.
        :rtype: Tensor
        """
        return self.effective_circuit.amplitude(l)

    def probability(self) -> Tensor:
        """
        Get the length-`2^n` probability vector over the computational basis.

        :return: Probability vector of shape `[dim^n]`.
        :rtype: Tensor
        """
        return self.effective_circuit.probability()

    def expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        ps: Optional[Sequence[int]] = None,
        reuse: bool = True,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        status: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Shortcut for Pauli string expectation.
        x, y, z list are for X, Y, Z positions

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.H(1)
        >>> c.expectation_ps(x=[1], z=[0])
        array(-0.99999994+0.j, dtype=complex64)

        :param x: sites to apply X gate, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: sites to apply Y gate, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: sites to apply Z gate, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param ps: or one can apply a ps structures instead of ``x``, ``y``, ``z``,
            e.g. [0, 1, 3, 0, 2, 2] for X_1Z_2Y_4Y_5
            defaults to None, ``ps`` can overwrite ``x``, ``y`` and ``z``
        :type ps: Optional[Sequence[int]], optional
        :param reuse: whether to cache and reuse the wavefunction, defaults to True
        :type reuse: bool, optional
        :param noise_conf: Noise Configuration, defaults to None
        :type noise_conf: Optional[NoiseConf], optional
        :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
        :type nmc: int, optional
        :param status: external randomness given by tensor uniformly from [0, 1], defaults to None,
            used for noisfy circuit sampling
        :type status: Optional[Tensor], optional
        :return: Expectation value
        :rtype: Tensor
        """
        return self.effective_circuit.expectation_ps(
            x=x,
            y=y,
            z=z,
            ps=ps,
            reuse=reuse,
            noise_conf=noise_conf,
            nmc=nmc,
            status=status,
            **kws,
        )

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
        r"""
        Batched sampling from the circuit or final state.

        :param batch: Number of samples. If `None`, returns a single draw.
        :type batch: Optional[int]
        :param allow_state: If `True`, sample from the final state (when memory allows). Prefer `True` for speed.
        :type allow_state: bool
        :param readout_error: Optional readout error model.
        :type readout_error: Optional[Sequence[Any]]
        :param format: Output format. See :py:meth:`tensorcircuit.quantum.measurement_results`.
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
        """
        return self.effective_circuit.sample(
            batch=batch,
            allow_state=allow_state,
            readout_error=readout_error,
            format=format,
            random_generator=random_generator,
            status=status,
            jittable=jittable,
        )

    def __repr__(self) -> str:
        s = f"AnalogCircuit(n={self.num_qubits}):\n"
        s += "=" * 40 + "\n"

        num_stages = len(self.analog_blocks) + 1

        for i in range(num_stages):
            # Print digital part
            s += f"--- Digital Block {i} ---\n"

            # Print analog part (if it exists)
            if i < len(self.analog_blocks):
                block = self.analog_blocks[i]
                s += f"--- Analog Block {i} (T={block.time}) ---\n"
                s += f" H(t) function: '{block.hamiltonian_func.__name__}'\n"

        s += "=" * 40
        return s
