"""End-to-end TianYan cloud example using the public TensorCircuit API.

Install TensorCircuit-NG and ``cqlib>=1.3.10,<1.4``, then set the
``TC_TOKEN_TIANYAN`` environment variable before running this script (the
cloud token system picks it up automatically). The simulator examples run by
default; pass ``--hardware`` to also opt into the real-device example.
"""

import argparse
import os

import tensorcircuit as tc

SIMULATOR = "tianyan_sw"
REAL_DEVICE = "tianyan176"


def _bell_circuit() -> tc.Circuit:
    circuit = tc.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_instruction(0, 1)
    return circuit


def main() -> None:
    """Run simulator workflows and optionally submit to real hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="also submit a task to the real device",
    )
    args = parser.parse_args()

    if not os.getenv("TC_TOKEN_TIANYAN"):
        raise RuntimeError("Set TC_TOKEN_TIANYAN before running this example")

    tc.cloud.apis.set_provider("tianyan")

    devices = tc.cloud.apis.list_devices(provider="tianyan")
    print(f"Available TianYan devices ({len(devices)}):")
    for device in devices:
        print(f"  - {device}")

    simulator = tc.cloud.apis.get_device(f"tianyan::{SIMULATOR}")
    circuit = _bell_circuit()
    task = tc.cloud.apis.submit_task(circuit=circuit, device=simulator, shots=1000)
    counts = task.results(blocked=True)
    details = task.details()

    print(f"\nSimulator task: {task.id_}")
    print(f"Counts: {counts}")
    print(f"State: {details['state']}")
    print(f"QCIS:\n{details['source']}")

    circuits = [_bell_circuit() for _ in range(3)]
    tasks = tc.cloud.apis.submit_task(circuit=circuits, device=simulator, shots=100)
    print("\nBatch results:")
    for index, batch_task in enumerate(tasks, start=1):
        print(f"  Task {index}: {batch_task.results(blocked=True)}")

    qcis = "H Q0\nH Q1\nCZ Q0 Q1\nH Q1\nM Q0\nM Q1"
    qcis_task = tc.cloud.apis.submit_task(source=qcis, device=simulator, shots=100)
    print(f"\nDirect QCIS result: {qcis_task.results(blocked=True)}")

    if not args.hardware:
        print("\nReal-device submission skipped. Pass --hardware to enable it.")
        return

    real_device = tc.cloud.apis.get_device(f"tianyan::{REAL_DEVICE}")
    # real hardware requires a circuit that already respects the device topology;
    # map the logical circuit onto a connected physical pair
    q1, q2 = sorted(real_device.topology()[0])
    hardware_circuit = circuit.initial_mapping({0: q1, 1: q2}, n=q2 + 1)
    real_task = tc.cloud.apis.submit_task(
        circuit=hardware_circuit, device=real_device, shots=100
    )
    real_counts = real_task.results(blocked=True)
    real_details = real_task.details()
    print(f"\nReal-device task: {real_task.id_}")
    print(f"Physical qubits: {q1}, {q2}")
    print(f"Counts: {real_counts}")
    print(f"QCIS:\n{real_details['source']}")


if __name__ == "__main__":
    main()
