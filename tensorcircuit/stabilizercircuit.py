"""
Stabilizer circuit simulator using Stim backend
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import stim

from .abstractcircuit import AbstractCircuit

Tensor = Any


class StabilizerCircuit(AbstractCircuit):
    """
    Quantum circuit simulator for stabilizer circuits using Stim backend.
    Supports Clifford operations and measurements.
    """

    # Add gate sets as class attributes
    clifford_gates = ["h", "x", "y", "z", "cnot", "cz", "swap", "s", "sd"]

    def __init__(self, nqubits: int) -> None:
        self._nqubits = nqubits
        self._stim_circuit = stim.Circuit()
        self._qir: List[Dict[str, Any]] = []
        self.is_dm = False
        self.inputs = None
        self._extra_qir: List[Dict[str, Any]] = []
        self.current_sim = stim.TableauSimulator()

    def apply_general_gate(
        self,
        gate: Any,
        *index: int,
        name: Optional[str] = None,
        **kws: Any,
    ) -> None:
        """
        Apply a Clifford gate to the circuit.

        :param gate: Gate to apply (must be Clifford)
        :type gate: Any
        :param index: Qubit indices to apply the gate to
        :type index: int
        :param name: Name of the gate operation, defaults to None
        :type name: Optional[str], optional
        :raises ValueError: If non-Clifford gate is applied
        """
        if name is None:
            name = ""

        # Record gate in QIR
        gate_dict = {
            "gate": gate,
            "index": index,
            "name": name,
            "split": kws.get("split", False),
            "mpo": kws.get("mpo", False),
        }
        ir_dict = kws.get("ir_dict", None)
        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)

        # Convert negative indices
        index = tuple([i if i >= 0 else self._nqubits + i for i in index])

        # Map TensorCircuit gates to Stim gates
        gate_map = {
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "cnot": "CNOT",
            "cz": "CZ",
            "swap": "SWAP",
            "s": "S",
            "sd": "S_DAG",
        }
        if name.lower() in gate_map:
            self._stim_circuit.append(gate_map[name.lower()], list(index))
            instruction = stim.Circuit()
            instruction.append(gate_map[name.lower()], list(index))
            self.current_sim.do(instruction)
        else:
            raise ValueError(f"Gate {name} is not supported in stabilizer simulation")

    apply = apply_general_gate

    def measure(self, *index: int, with_prob: bool = False) -> Tensor:
        """
        Measure qubits in Z basis.

        :param index: Indices of qubits to measure
        :type index: int
        :param with_prob: Return probability of measurement outcome, defaults to False
        :type with_prob: bool, optional
        :return: Measurement results and probability (if with_prob=True)
        :rtype: Union[np.ndarray, Tuple[np.ndarray, float]]
        """
        # Convert negative indices

        index = tuple([i for i in index if i >= 0])

        # Add measurement instructions
        s1 = self.current_simulator().copy()
        m = s1.measure_many(*index)
        # Sample once from the circuit using sampler

        # TODO(@refraction-ray): correct probability
        return m

    def cond_measurement(self, index: int) -> Tensor:
        """
        Measure qubits in Z basis with state collapse.

        :param index: Index of qubit to measure
        :type index: int
        :return: Measurement results and probability (if with_prob=True)
        :rtype: Union[np.ndarray, Tuple[np.ndarray, float]]
        """
        # Convert negative indices

        # Add measurement instructions
        self._stim_circuit.append("M", index)
        m = self.current_simulator().measure(index)
        # Sample once from the circuit using sampler

        return m

    cond_measure = cond_measurement

    def sample(
        self,
        batch: Optional[int] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Sample measurements from the circuit.

        :param batch: Number of samples to take, defaults to None (single sample)
        :type batch: Optional[int], optional
        :return: Measurement results
        :rtype: Tensor
        """
        if batch is None:
            batch = 1
        c = self.current_circuit().copy()
        for i in range(self._nqubits):
            c.append("M", [i])
        sampler = c.compile_sampler()
        samples = sampler.sample(batch)
        return np.array(samples)

    def expectation_ps(  # type: ignore
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        **kws: Any,
    ) -> Any:
        """
        Compute exact expectation value of Pauli string using stim's direct calculation.

        :param x: Indices for Pauli X measurements
        :type x: Optional[Sequence[int]], optional
        :param y: Indices for Pauli Y measurements
        :type y: Optional[Sequence[int]], optional
        :param z: Indices for Pauli Z measurements
        :type z: Optional[Sequence[int]], optional
        :return: Expectation value
        :rtype: float
        """
        # Build Pauli string representation
        pauli_str = ["I"] * self._nqubits

        if x:
            for i in x:
                pauli_str[i] = "X"
        if y:
            for i in y:
                pauli_str[i] = "Y"
        if z:
            for i in z:
                pauli_str[i] = "Z"

        pauli_string = "".join(pauli_str)
        # Calculate expectation using stim's direct method
        expectation = self.current_simulator().peek_observable_expectation(
            stim.PauliString(pauli_string)
        )
        return expectation

    expps = expectation_ps

    def sample_expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        shots: Optional[int] = None,
        **kws: Any,
    ) -> float:
        """
        Compute expectation value of Pauli string measurements.

        :param x: Indices for Pauli X measurements, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: Indices for Pauli Y measurements, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: Indices for Pauli Z measurements, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param shots: Number of measurement shots, defaults to None
        :type shots: Optional[int], optional
        :return: Expectation value
        :rtype: float
        """
        if shots is None:
            shots = 1000  # Default number of shots

        circuit = self._stim_circuit.copy()

        # Add basis rotations for measurements
        if x:
            for i in x:
                circuit.append("H", [i])
        if y:
            for i in y:
                circuit.append("S_DAG", [i])
                circuit.append("H", [i])

        # Add measurements
        measured_qubits: List[int] = []
        if x:
            measured_qubits.extend(x)
        if y:
            measured_qubits.extend(y)
        if z:
            measured_qubits.extend(z)

        for i in measured_qubits:
            circuit.append("M", [i])

        # Sample and compute expectation using sampler
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots)
        results = np.array(samples)

        # Convert from {0,1} to {1,-1}
        results = 1 - 2 * results

        # Average over shots
        expectation = np.mean(np.prod(results, axis=1))

        return float(expectation)

    sexpps = sample_expectation_ps

    def mid_measurement(self, index: int, keep: int = 0) -> Tensor:
        """
        Perform a mid-measurement operation on a qubit on z direction.
        The post-selection cannot be recorded in ``stim.Circuit``

        :param index: Index of the qubit to measure
        :type index: int
        :param keep: State of qubits to keep after measurement, defaults to 0 (up)
        :type keep: int, optional
        :return: Result of the mid-measurement operation
        :rtype: Tensor
        """
        if keep not in [0, 1]:
            raise ValueError("keep must be 0 or 1")

        self.current_sim.postselect_z(index, desired_value=keep)

    mid_measure = mid_measurement
    post_select = mid_measurement
    post_selection = mid_measurement

    def current_simulator(self) -> stim.TableauSimulator:
        """
        Return the current simulator of the circuit.
        """
        return self.current_sim

    def current_circuit(self) -> stim.Circuit:
        """
        Return the current stim circuit representation of the circuit.
        """
        return self._stim_circuit

    def current_tableau(self) -> stim.Tableau:
        """
        Return the current tableau of the circuit.
        """
        self.current_simulator().current_inverse_tableau() ** -1


# Call _meta_apply at module level to register the gates
StabilizerCircuit._meta_apply()
