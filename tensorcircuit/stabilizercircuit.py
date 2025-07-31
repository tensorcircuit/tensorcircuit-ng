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

    def __init__(
        self, nqubits: int, inputs: Tensor = None, tableau_inputs: Tensor = None
    ) -> None:
        """
        ``StabilizerCircuit`` class based on stim package

        :param nqubits: Number of qubits
        :type nqubits: int
        :param inputs: initial state by stabilizers, defaults to None
        :type inputs: Tensor, optional
        :param tableau_inputs: initial state by **inverse** tableau, defaults to None
        :type tableau_inputs: Tensor, optional
        """
        self._nqubits = nqubits
        self._stim_circuit = stim.Circuit()
        self._qir: List[Dict[str, Any]] = []
        self.is_dm = False
        self.inputs = None
        self._extra_qir: List[Dict[str, Any]] = []
        self.current_sim = stim.TableauSimulator()
        if inputs:
            self.current_sim.set_state_from_stabilizers(inputs)
        if tableau_inputs:
            self.current_sim.set_inverse_tableau(tableau_inputs)

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
            "split": None,
            "mpo": False,
        }
        ir_dict = kws["ir_dict"]
        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)

        # Map TensorCircuit gates to Stim gates

        if name.lower() in self.gate_map:
            # self._stim_circuit.append(gate_map[name.lower()], list(index))
            gn = self.gate_map[name.lower()]
            instruction = f"{gn} {' '.join(map(str, index))}"
            self._stim_circuit.append_from_stim_program_text(instruction)
            # append is much slower
            # self.current_sim.do(stim.Circuit(instruction))
            getattr(self.current_sim, gn.lower())(*index)
        else:
            raise ValueError(f"Gate {name} is not supported in stabilizer simulation")

    apply = apply_general_gate

    def state(self) -> Tensor:
        """
        Return the wavefunction of the circuit.
        Note that the state can have smaller qubit count if no gate is applied on later qubits
        """
        tab = self.current_tableau()
        return tab.to_state_vector(endian="big")

    def random_gate(self, *index: int, recorded: bool = False) -> None:
        """
        Apply a random Clifford gate to the circuit.
        This operation will not record in qir

        :param index: Qubit indices to apply the gate to
        :type index: int
        :param recorded: Whether the gate is recorded in ``stim.Circuit``, defaults to False
        :type recorded: bool, optional
        """
        m = len(index)
        t = stim.Tableau.random(m)
        self.current_sim.do_tableau(t, index)
        if recorded:
            self._stim_circuit += t.to_circuit()

    def tableau_gate(self, *index: int, tableau: Any, recorded: bool = False) -> None:
        """
        Apply a gate indicated by tableau to the circuit.
        This operation will not record in qir

        :param index: Qubit indices to apply the gate to
        :type index: int
        :param tableau: stim.Tableau representation of the gate
        :type tableau: Any
        :param recorded: Whether the gate is recorded in ``stim.Circuit``, defaults to False
        :type recorded: bool, optional
        """
        self.current_sim.do_tableau(tableau, index)
        if recorded:
            self._stim_circuit += tableau.to_circuit()

    def measure(self, *index: int, with_prob: bool = False) -> Tensor:
        """
        Measure qubits in the Z basis.

        :param index: Indices of the qubits to measure.
        :type index: int
        :param with_prob: If True, returns the theoretical probability of the measurement outcome.
            defaults to False
        :type with_prob: bool, optional
        :return: A tensor containing the measurement results.
            If `with_prob` is True, a tuple containing the results and the probability is returned.
        :rtype: Tensor
        """
        # Convert negative indices

        index = tuple([i for i in index if i >= 0])

        # Add measurement instructions
        s1 = self.current_simulator().copy()
        # Sample once from the circuit using sampler

        if with_prob:
            num_random_measurements = 0
            for i in index:
                if s1.peek_z(i) == 0:
                    num_random_measurements += 1
            probability = (0.5) ** num_random_measurements

        m = s1.measure_many(*index)
        if with_prob:
            return m, probability
        return m

    def cond_measurement(self, index: int) -> Tensor:
        """
        Measure a single qubit in the Z basis and collapse the state.

        :param index: The index of the qubit to measure.
        :type index: int
        :return: The measurement result (0 or 1).
        :rtype: Tensor
        """
        # Convert negative indices

        # Add measurement instructions
        self._stim_circuit.append_from_stim_program_text("M " + str(index))
        # self.current_sim = None
        m = self.current_simulator().measure(index)
        # Sample once from the circuit using sampler

        return m

    cond_measure = cond_measurement

    def cond_measure_many(self, *index: int) -> Tensor:
        """
        Measure multiple qubits in the Z basis and collapse the state.

        :param index: The indices of the qubits to measure.
        :type index: int
        :return: A tensor containing the measurement results.
        :rtype: Tensor
        """
        # Convert negative indices

        # Add measurement instructions
        self._stim_circuit.append_from_stim_program_text(
            "M " + " ".join(map(str, index))
        )
        # self.current_sim = None
        m = self.current_simulator().measure_many(*index)
        # Sample once from the circuit using sampler

        return m

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

    def depolarizing(self, *index: int, p: float) -> None:
        """
        Apply depolarizing noise to a qubit.

        :param index: Index of the qubit to apply noise to
        :type index: int
        :param p: Noise parameter (probability of depolarizing)
        :type p: float
        """
        self._stim_circuit.append_from_stim_program_text(
            f"DEPOLARIZE1({p}) {' '.join(map(str, index))}"
        )
        self.current_sim.depolarize1(*index, p=p)

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
        return self.current_simulator().current_inverse_tableau() ** -1

    def current_inverse_tableau(self) -> stim.Tableau:
        """
        Return the current inverse tableau of the circuit.
        """
        return self.current_simulator().current_inverse_tableau()

    def entanglement_entropy(self, cut: Sequence[int]) -> float:
        """
        Calculate the entanglement entropy for a subset of qubits using stabilizer formalism.

        :param cut: Indices of qubits to calculate entanglement entropy for
        :type cut: Sequence[int]
        :return: Entanglement entropy
        :rtype: float
        """
        # Get stabilizer tableau
        tableau = self.current_tableau()
        N = len(tableau)

        # Pre-allocate binary matrix with proper dtype
        # binary_matrix = np.zeros((N, 2 * N), dtype=np.int8)

        # Vectorized conversion of stabilizers to binary matrix
        # z_outputs = np.array([tableau.z_output(k) for k in range(N)])
        # x_part = z_outputs == 1  # X
        # z_part = z_outputs == 3  # Z
        # y_part = z_outputs == 2  # Y

        # binary_matrix[:, :N] = x_part | y_part
        # binary_matrix[:, N:] = z_part | y_part

        _, _, z2x, z2z, _, _ = tableau.to_numpy()
        binary_matrix = np.concatenate([z2x, z2z], axis=1)
        # Get reduced matrix for the cut using boolean indexing
        cut_set = set(cut)
        cut_indices = np.array(
            [i for i in range(N) if i in cut_set]
            + [i + N for i in range(N) if i in cut_set]
        )
        reduced_matrix = binary_matrix[:, cut_indices]

        # Efficient rank calculation using Gaussian elimination
        matrix = reduced_matrix.copy()
        n_rows, n_cols = matrix.shape
        rank = 0
        row = 0

        for col in range(n_cols):
            # Vectorized pivot finding
            pivot_rows = np.nonzero(matrix[row:, col])[0]
            if len(pivot_rows) > 0:
                pivot_row = pivot_rows[0] + row

                # Swap rows if necessary
                if pivot_row != row:
                    matrix[row], matrix[pivot_row] = (
                        matrix[pivot_row].copy(),
                        matrix[row].copy(),
                    )

                # Vectorized elimination
                eliminate_mask = matrix[row + 1 :, col] == 1
                matrix[row + 1 :][eliminate_mask] ^= matrix[row]

                rank += 1
                row += 1

                if row == n_rows:
                    break

        # Calculate entropy
        return float((rank - len(cut)) * np.log(2))


# Call _meta_apply at module level to register the gates
StabilizerCircuit._meta_apply()
