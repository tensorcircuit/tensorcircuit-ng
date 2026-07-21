# TianYan Quantum Cloud

## Overview

TensorCircuit-NG integrates with the China Telecom TianYan quantum computing platform through the official `cqlib` SDK. The provider uses the shared `tc.cloud.apis` interface and supports simulator jobs, QCIS/OpenQASM submission, task inspection, batch submission, and automatic topology mapping for real devices.

## Platform features

- **Provider**: `tianyan`
- **SDK**: `cqlib` (official TianYan Python SDK)
- **Native instruction set**: QCIS
- **Backends**: software simulators and superconducting quantum hardware

## Quick start

### 1. Install dependencies

```bash
pip install "tensorcircuit-ng[cloud]"
pip install "cqlib>=1.3.10,<1.4"
```

The verified `cqlib` range requires Python 3.10 or later and NumPy 2.1.2 or later. `cqlib` is installed separately because it is not part of TensorCircuit-NG's generic `cloud` extra.

### 2. Configure the login key

Register at the [TianYan quantum cloud platform](https://qc.zdxlz.com) and obtain an SDK login key. Do not put the key in source files, notebooks, logs, or version control.

On Windows PowerShell, store it in the active Conda environment:

```powershell
conda activate <env>
conda env config vars set LOGIN_KEY="your_login_key"
conda deactivate
conda activate <env>
```

There must be no spaces around `=`. Reactivate the environment before using the variable.

### 3. Configure the provider and submit a task

```python
import os

import tensorcircuit as tc

login_key = os.getenv("LOGIN_KEY")
if not login_key:
    raise RuntimeError("Set LOGIN_KEY before using TianYan")

tc.cloud.apis.set_provider("tianyan")
tc.cloud.apis.set_token(login_key, provider="tianyan", cached=False)

devices = tc.cloud.apis.list_devices(provider="tianyan")
print(devices)

device = tc.cloud.apis.get_device("tianyan::tianyan_sw")
circuit = tc.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_instruction(0, 1)

task = tc.cloud.apis.submit_task(circuit=circuit, device=device, shots=1000)
print(task.results(blocked=True))
```

## API reference

### Devices and properties

| API | Description |
|---|---|
| `tc.cloud.apis.set_provider("tianyan")` | Select the TianYan provider |
| `tc.cloud.apis.set_token(key, provider="tianyan")` | Set the login key |
| `tc.cloud.apis.list_devices()` | List devices returned by the platform |
| `tc.cloud.apis.get_device("tianyan::DEVICE_NAME")` | Construct a device object |
| `device.list_properties()` | Read topology and calibration properties |
| `device.topology()` | Return the coupling edges |
| `device.native_gates()` | Return the supported native gates |

### Task submission

| API | Description |
|---|---|
| `tc.cloud.apis.submit_task(circuit=c, device=d, shots=1000)` | Submit one TensorCircuit circuit |
| `tc.cloud.apis.submit_task(circuit=[c1, c2], ...)` | Submit a batch of circuits |
| `tc.cloud.apis.submit_task(source="QCIS string", ...)` | Submit QCIS directly |
| `tc.cloud.apis.submit_task(source=qasm, lang="OPENQASM", ...)` | Convert and submit OpenQASM 2 |

### Task management

| API | Description |
|---|---|
| `task.results(blocked=True)` | Wait for and return measurement counts |
| `task.details()` | Return state, source, mapping, shots, and results |
| `task.state()` | Query the normalized task state |
| `task.resubmit()` | Submit the task again |
| `tc.cloud.apis.get_task(task_id, device=device)` | Reconstruct a task object by ID |

The current `cqlib` release does not expose a usable task-listing or cancellation endpoint. Therefore `tc.cloud.apis.list_tasks(...)` and `tc.cloud.apis.remove_task(...)` raise `NotImplementedError` for TianYan.

## Cloud simulator

The simulator has no hardware coupling constraints and is the recommended backend for developing a circuit. Batch submission returns one `Task` object per circuit:

```python
circuits = [c1, c2, c3]
tasks = tc.cloud.apis.submit_task(
    circuit=circuits,
    device=device,
    shots=100,
)
for task in tasks:
    print(task.results(blocked=True))
```

QCIS can be submitted without constructing a TensorCircuit circuit:

```python
qcis = "H Q0\nH Q1\nCZ Q0 Q1\nH Q1\nM Q0\nM Q1"
task = tc.cloud.apis.submit_task(source=qcis, device=device, shots=100)
print(task.results(blocked=True))
```

OpenQASM 2 is converted through `cqlib` before submission:

```python
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""
task = tc.cloud.apis.submit_task(
    source=qasm,
    lang="OPENQASM",
    device=device,
    shots=100,
)
```

## Compilation and topology mapping

When a TensorCircuit circuit is submitted to real hardware, TianYan automatically:

1. Queries the device topology.
2. Finds physical qubits that satisfy the circuit interactions.
3. Maps logical qubits to physical qubits.
4. Produces the corresponding QCIS source.

The mapping and generated source are available in `task.details()`:

```python
run_hardware = os.getenv("TIANYAN_RUN_HARDWARE") == "1"
if run_hardware:
    hardware_device = tc.cloud.apis.get_device("tianyan::tianyan176")
    hardware_task = tc.cloud.apis.submit_task(
        circuit=circuit,
        device=hardware_device,
        shots=100,
    )
    details = hardware_task.details()
    print(details["mapping"])
    print(details["source"])
else:
    print("Real-device submission skipped")
```

## Device list

The following names are examples only; use `list_devices()` for the live list and status.

| Device | Type | Description |
|---|---|---|
| `tianyan_sw` | Simulator | Full-state simulator |
| `tianyan_s` | Simulator | Single-amplitude simulator |
| `tianyan_tn` | Simulator | Tensor-network simulator |
| `tianyan176` | Hardware | 176-qubit superconducting device |
| `tianyan24` | Hardware | 24-qubit superconducting device |

## Caveats

1. Simulators are not subject to hardware coupling constraints; real devices have connectivity limits and noise.
2. Automatic topology mapping is enabled for real devices only.
3. Some hardware backends may reject circuits that use unsupported single-qubit paths or gates.
4. Keep hardware shots small to avoid unnecessary resource consumption.
5. The example and notebook skip hardware by default. Set `TIANYAN_RUN_HARDWARE=1` explicitly to enable it.
6. Measurements are submitted as terminal measurements in their recorded order. Mid-circuit measurement semantics are not preserved; do not submit circuits that depend on them.

## Troubleshooting

### `cqlib` is not installed

```bash
pip install "cqlib>=1.3.10,<1.4"
```

### Invalid login key

Check that `LOGIN_KEY` is visible in the active environment and obtain a fresh key from the [TianYan platform](https://qc.zdxlz.com) if needed.

### Topology mapping fails

- Check that the device has enough connected available qubits.
- Inspect `device.list_properties()["links"]`.
- Simplify the circuit and reduce multi-qubit gate interactions.

### Task execution fails

- Check that the selected device supports the circuit's gates and interactions.
- Check the live device state returned by the platform.

## Related files

- `tensorcircuit/cloud/tianyan.py`: TianYan provider implementation
- `tensorcircuit/cloud/apis.py`: shared cloud API entry point
- `examples/tianyan_cloud_demo.py`: end-to-end Python example
- `docs/source/tutorials/tianyan_cloud.ipynb`: English Jupyter tutorial
- `docs/source/tutorials/tianyan_cloud_reference.md`: this reference document

## More information

- [TianYan platform documentation](https://cqlib.readthedocs.io/)
- [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/)
