"""Independent full-space checks for the entanglement-growth example."""

from itertools import combinations
from math import comb

import numpy as np
from scipy.linalg import expm
from main import (
    Evolution,
    Floquet,
    JaxKernels,
    PROTOCOLS,
    gate,
    hamiltonian,
    sector_indices,
)

import tensorcircuit as tc


def full_entropy(state, length, cut, eps=0.0):
    """Full-Hilbert-space SVD reference, independent of sector/cut index maps."""
    rest = tuple(site for site in range(length) if site not in cut)
    matrix = state.reshape([2] * length).transpose(tuple(cut) + rest)
    singular = np.linalg.svd(matrix.reshape(2 ** len(cut), -1), compute_uv=False)
    probabilities = singular**2 / np.sum(np.abs(state) ** 2)
    probabilities = (probabilities + eps) / (1 + len(probabilities) * eps)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities + eps))


def full_gate(states, unitary, bond, length):
    """Apply a gate to full-space tensor axes, with states in columns."""
    tensor = states.reshape(2**bond, 4, 2 ** (length - bond - 2), -1)
    return np.einsum("ab,lbrs->lars", unitary, tensor).reshape(states.shape)


def verify_physics(length):
    rng = np.random.default_rng(91)
    indices = sector_indices(length)
    psi = rng.normal(size=len(indices)) + 1j * rng.normal(size=len(indices))
    psi /= np.linalg.norm(psi)
    spin = [
        np.array([[0, 1], [1, 0]]) / 2,
        np.array([[0, -1j], [1j, 0]]) / 2,
        np.diag([1, -1]) / 2,
    ]
    fields = rng.uniform(-5, 5, length)

    def product(operators):
        result = np.ones((1, 1))
        for site in range(length):
            result = np.kron(result, operators.get(site, np.eye(2)))
        return result

    for jz in (0, 0.5, 1):
        reference = np.zeros((2**length, 2**length), dtype=complex)
        for site in range(length - 1):
            for operator, weight in zip(spin, (1, 1, jz)):
                reference += weight * product({site: operator, site + 1: operator})
        for site, field in enumerate(fields):
            reference += field * product({site: spin[2]})
        np.testing.assert_allclose(
            hamiltonian(length, fields, jz),
            reference[np.ix_(indices, indices)],
            atol=1e-13,
        )
    matrix = hamiltonian(length, fields)
    evolution = Evolution(matrix)
    for time in (0, 0.25, 4.5, 32):
        np.testing.assert_allclose(
            next(evolution.at(psi, [time])), expm(-1j * time * matrix) @ psi, atol=2e-12
        )
    np.testing.assert_allclose(np.linalg.norm(next(evolution.at(psi, [1e15]))), 1)
    for alpha, beta in [*PROTOCOLS.values(), (np.pi, np.pi)]:
        expected = expm(
            -1j
            * (
                alpha * (np.kron(spin[0], spin[0]) + np.kron(spin[1], spin[1]))
                + beta * np.kron(spin[2], spin[2])
            )
        )
        np.testing.assert_allclose(gate(alpha, beta), expected, atol=1e-13)
    swap = np.eye(4)[[0, 2, 1, 3]]
    np.testing.assert_allclose(
        gate(np.pi, np.pi), np.exp(-1j * np.pi / 4) * swap, atol=1e-13
    )
    floquet = Floquet(length, fields)
    expected = expm(-1j * hamiltonian(length, fields, jz=1, jxy=0)) @ expm(
        -0.4j * hamiltonian(length, np.zeros(length), jz=0)
    )
    for period in (0, 1, 3, 10):
        np.testing.assert_allclose(
            next(floquet.at(psi, [period])),
            np.linalg.matrix_power(expected, period) @ psi,
            atol=2e-12,
        )
    print(f"L={length}: passed XXZ, evolution, gate exponentials, and Floquet checks.")


def verify_kernels(length):
    """Compare JAX batches and scans with full-space SVD and TC circuits."""
    rng = np.random.default_rng(671)
    kernels = JaxKernels(length)
    indices = sector_indices(length)
    states = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(
        size=(len(indices), 3)
    )
    states[:, 0] = 0
    states[0, 0] = 1
    states /= np.linalg.norm(states, axis=0)
    full = np.zeros((2**length, 3), dtype=complex)
    full[indices] = states
    half = tuple(range(length // 2))
    np.testing.assert_allclose(
        kernels.hcee(states),
        [full_entropy(s, length, half) for s in full.T],
        atol=2e-12,
    )
    # Every cut, including complements; the generic TC helper uses natural logs.
    K = kernels.K
    entropy_batch = K.jit(K.vmap(kernels.entropy_at, vectorized_argnums=0))
    for cut in combinations(range(length), length // 2):
        actual = K.numpy(
            entropy_batch(K.convert_to_tensor(states.T), kernels.cut_maps(cut))
        )
        reference = [full_entropy(s, length, cut) for s in full.T]
        np.testing.assert_allclose(actual, reference, atol=2e-12)
        # TC's generic helper regularizes both the density matrix and logarithm.
        np.testing.assert_allclose(
            [full_entropy(s, length, cut, eps=1e-12) for s in full.T],
            [
                tc.quantum.entanglement_entropy(s, subsystem_to_keep=cut) / np.log(2)
                for s in full.T
            ],
            atol=2e-12,
            rtol=0,
        )
        rest = tuple(site for site in range(length) if site not in cut)
        np.testing.assert_allclose(
            actual, [full_entropy(s, length, rest) for s in full.T], atol=2e-12
        )
    assert len(kernels.cuts) == comb(length, length // 2) // 2
    np.testing.assert_allclose(
        kernels.baee(states),
        np.mean(
            [[full_entropy(s, length, cut) for s in full.T] for cut in kernels.cuts],
            axis=0,
        ),
        atol=2e-12,
    )
    before = kernels.baee(states[:, 1:2])
    state = full[:, 1].copy()
    for bond in rng.permutation(length - 1):
        circuit = tc.Circuit(length, inputs=state)
        circuit.any(int(bond), int(bond) + 1, unitary=gate(np.pi, np.pi))
        state = circuit.state()
        np.testing.assert_allclose(
            kernels.baee(state[indices, None]), before, atol=2e-12
        )

    unitaries = [gate(a, b) for a, b in [*PROTOCOLS.values(), (np.pi, np.pi)]]
    # A non-symmetric charge-conserving gate catches accidental transposition.
    circuit = tc.Circuit(2)
    circuit.rz(0, theta=0.37)
    circuit.any(0, 1, unitary=gate(np.pi / 2, np.pi / 3))
    unitaries.append(circuit.matrix())
    # 107 layers exercise both the warmup and final 100-layer measurement scan.
    bonds = rng.integers(0, length - 1, (2, 107))
    for unitary in unitaries:
        for bond in range(length - 1):
            expected = full_gate(full, unitary, bond, length)
            for column, initial in enumerate(full.T):
                circuit = tc.Circuit(length, inputs=initial)
                circuit.any(bond, bond + 1, unitary=unitary)
                np.testing.assert_allclose(
                    expected[:, column], circuit.state(), atol=2e-12
                )
            # Also check the short-circuit branch, where the warmup is empty.
            final, mean = kernels.circuit(
                K.convert_to_tensor(states.T),
                K.convert_to_tensor(unitary),
                K.convert_to_tensor(np.array([[bond]])),
            )
            np.testing.assert_allclose(
                K.numpy(final[0]), expected[indices].T, atol=2e-12
            )
            np.testing.assert_allclose(
                K.numpy(mean[0]),
                [full_entropy(s, length, half) for s in expected.T],
                atol=2e-12,
            )
        finals, means = kernels.circuit(
            K.convert_to_tensor(states.T),
            K.convert_to_tensor(unitary),
            K.convert_to_tensor(bonds),
        )
        for realization, sequence in enumerate(bonds):
            state, total = full.copy(), np.zeros(states.shape[1])
            for depth, bond in enumerate(sequence):
                state = full_gate(state, unitary, bond, length)
                if depth >= len(sequence) - 100:
                    total += [full_entropy(s, length, half) for s in state.T]
            np.testing.assert_allclose(
                K.numpy(finals[realization]), state[indices].T, atol=2e-12, rtol=1e-11
            )
            np.testing.assert_allclose(
                K.numpy(means[realization]), total / 100, atol=2e-12, rtol=1e-11
            )
        np.testing.assert_allclose(
            kernels.rqc(states, unitary, bonds), K.numpy(means).mean(axis=0), atol=2e-12
        )
    if length == 4:
        bell = np.zeros((len(indices), 1), dtype=complex)
        bell[np.isin(indices, [0b0101, 0b0011]), 0] = 1 / np.sqrt(2)
        np.testing.assert_allclose(kernels.hcee(bell), [1], atol=2e-12)
    print(
        f"L={length}: passed JAX batches, every cut, TC gates, SWAP, and circuit scans."
    )


if __name__ == "__main__":
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        for size in (4, 6):
            verify_physics(size)
        for size in (4, 6, 8, 12):
            verify_kernels(size)
