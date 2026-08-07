r"""Magic growth during real-time mixed-field Ising dynamics.

The example uses exact diagonalization to obtain state snapshots and then
estimates the qubit stabilizer Rényi entropy ``M_2``.  When the raw estimate
has a large Monte Carlo error bar, a fixed layer of Hadamard gates is tried on
the snapshot.  The preconditioned result is kept only when its error bar is
smaller.
"""

import matplotlib.pyplot as plt
import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def mixed_field_ising_hamiltonian(n: int):
    """Return an open-chain mixed-field Ising Hamiltonian."""
    hx = (np.sqrt(5.0) + 5.0) / 8.0
    hz = (np.sqrt(5.0) + 1.0) / 4.0
    graph = tc.templates.graphs.Line1D(n, pbc=False)
    return tc.quantum.heisenberg_hamiltonian(
        graph, hzz=1.0, hxx=0.0, hyy=0.0, hx=hx, hz=hz, sparse=False
    )


def neel_state(n: int):
    """Return ``|0101...>`` as a wavefunction."""
    circuit = tc.Circuit(n)
    circuit.x(list(range(1, n, 2)))
    return circuit.state()


def apply_hadamards(state, nqubits, qubits):
    """Apply a fixed Hadamard layer to a state snapshot."""
    circuit = tc.Circuit(nqubits, inputs=state)
    for qubit in qubits:
        circuit.h(qubit)
    return circuit.state()


def main() -> None:
    """Evolve the state, estimate ``M_2``, and save a simple trajectory plot."""
    n = 8
    times = np.linspace(0.0, 8.0, 33)
    samples = 512
    error_threshold = 0.12
    preconditioner_qubits = tuple(range(n))

    hamiltonian = mixed_field_ising_hamiltonian(n)
    psi0 = neel_state(n)
    states = tc.timeevol.ed_evol(
        hamiltonian,
        psi0,
        1.0j * K.convert_to_tensor(times),
    )

    @K.jit
    def estimate(state, status):
        return tc.quantum.stabilizer_renyi_entropy(
            state, alpha=2, status=status, with_std=True
        )

    @K.jit
    def estimate_preconditioned(state, status):
        preconditioned_state = apply_hadamards(state, n, preconditioner_qubits)
        return estimate(preconditioned_state, status)

    rng = np.random.default_rng(1234)
    status_values = rng.random((len(times), samples)).astype(np.float32)
    magic = []
    error = []
    used_preconditioner = []

    for state, status_value in zip(states, status_values):
        status = K.convert_to_tensor(status_value)
        value, std = estimate(state, status)
        value = float(K.numpy(value))
        std = float(K.numpy(std))
        used = False

        if std > error_threshold:
            pre_value, pre_std = estimate_preconditioned(state, status)
            pre_value = float(K.numpy(pre_value))
            pre_std = float(K.numpy(pre_std))
            if pre_std < std:
                value, std = pre_value, pre_std
                used = True

        magic.append(value)
        error.append(std)
        used_preconditioner.append(used)

    magic = np.asarray(magic)
    error = np.asarray(error)
    used_preconditioner = np.asarray(used_preconditioner)

    plt.errorbar(
        times,
        magic,
        yerr=error,
        fmt="o-",
        capsize=2,
        label="selected estimate",
    )
    if np.any(used_preconditioner):
        plt.scatter(
            times[used_preconditioner],
            magic[used_preconditioner],
            color="C3",
            marker="s",
            s=36,
            zorder=3,
            label="Hadamard preconditioned",
        )
    plt.axhline(np.log2(2**n + 3.0) - 2.0, color="k", linestyle=":", label="Haar")
    plt.xlabel("Time")
    plt.ylabel(r"Stabilizer Rényi entropy $M_2$")
    plt.legend()
    plt.tight_layout()

    output_path = "examples/mixed_field_ising_magic_dynamics.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
