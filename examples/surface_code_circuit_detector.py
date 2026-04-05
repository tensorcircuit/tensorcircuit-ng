"""
Distance-3 detector sampling benchmark using tc.Circuit tensor network path.
Modified to include multiple rounds, depolarizing noise and explicit status feeding.
"""

import time
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_contractor("cotengra")


def build_surface_code_d3(
    is_dm: bool, error_prob: float = 0.0, rounds: int = 1
) -> tc.Circuit:
    # 9 data qubits, 8 ancilla qubits (4 Z checks + 4 X checks)
    z_checks = [
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8],
    ]
    x_checks = [
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8],
    ]

    n_data = 9
    n_anc = 8
    if is_dm:
        c = tc.DMCircuit(n_data + n_anc)
    else:
        c = tc.Circuit(n_data + n_anc)

    # Isotropic depolarizing parameters
    px = py = pz = error_prob / 3.0

    for r in range(rounds):
        # 1. Syndrome extraction (Interactions)
        for a, support in enumerate(z_checks):
            anc = n_data + a
            for q in support:
                c.cx(q, anc)
                if error_prob > 0:
                    c.depolarizing(q, px=px, py=py, pz=pz)
                    c.depolarizing(anc, px=px, py=py, pz=pz)

        for a, support in enumerate(x_checks):
            anc = n_data + 4 + a
            c.h(anc)
            for q in support:
                c.cx(anc, q)
                if error_prob > 0:
                    c.depolarizing(anc, px=px, py=py, pz=pz)
                    c.depolarizing(q, px=px, py=py, pz=pz)
            c.h(anc)

        # 2. Measurement and Detector definition
        for a in range(n_anc):
            anc = n_data + a
            c.measure_instruction(anc)
            if r == 0:
                # First round detector is just the syndrome
                c.detector_instruction([-1])
            else:
                # Subsequent round detector is XOR of current and previous syndrome
                # Previous measurement for this specific ancilla was n_anc indices ago
                c.detector_instruction([-1, -(n_anc + 1)])

            # Reset ancilla for next round if needed
            if r < rounds - 1:
                c.reset_instruction(anc)

    return c


def main() -> None:
    shots = 50000
    batch = 1000
    error_prob = 0.005
    rounds = 1

    print(f"--- Surface Code Benchmark (d=3, rounds={rounds}, error={error_prob}) ---")

    print(f"DMCircuit+sampling (allow_state=False)")
    c = build_surface_code_d3(True, error_prob=error_prob, rounds=rounds)
    num_detectors = 8 * rounds
    status = tc.backend.implicit_randu(shape=[shots, num_detectors])
    t0 = time.time()
    samples = c.sample_detector(shots=shots, batch=batch, status=status)
    t1 = time.time()
    print(f"detectors={samples.shape[1]}")
    print(f"shots={samples.shape[0]}")
    print(f"mean={np.mean(samples):.6f}")
    print(f"time={t1 - t0:.4f}s")

    print("\nCircuit with trajectory+sampling (allow_state=False)")
    c = build_surface_code_d3(False, error_prob=error_prob, rounds=rounds)
    qir = c._merge_qir_with_extra()
    num_events = c._count_detector_random_events(qir)
    print(f"num_random_events={num_events}")
    status = tc.backend.implicit_randu(shape=[shots, num_events])
    t0 = time.time()
    samples = c.sample_detector(shots=shots, batch=batch, status=status)
    t1 = time.time()
    print(f"detectors={samples.shape[1]}")
    print(f"shots={samples.shape[0]}")
    print(f"mean={np.mean(samples):.6f}")
    print(f"time={t1 - t0:.4f}s")

    print("\nDMCircuit+full-state (allow_state=True)")
    c = build_surface_code_d3(True, error_prob=error_prob, rounds=rounds)
    status = tc.backend.implicit_randu(shape=[shots])
    t0 = time.time()
    samples = c.sample_detector(
        shots=shots, batch=batch, allow_state=True, status=status
    )
    t1 = time.time()
    print(f"detectors={samples.shape[1]}")
    print(f"shots={samples.shape[0]}")
    print(f"mean={np.mean(samples):.6f}")
    print(f"time={t1 - t0:.4f}s")

    print("\nCircuit+full-state (allow_state=True)")
    c = build_surface_code_d3(False, error_prob=error_prob, rounds=rounds)
    status = tc.backend.implicit_randu(shape=[shots])
    t0 = time.time()
    samples = c.sample_detector(
        shots=shots, batch=batch, allow_state=True, status=status
    )
    t1 = time.time()
    print(f"detectors={samples.shape[1]}")
    print(f"shots={samples.shape[0]}")
    print(f"mean={np.mean(samples):.6f}")
    print(f"time={t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
