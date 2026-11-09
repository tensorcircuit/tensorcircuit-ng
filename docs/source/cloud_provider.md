# Adding a Cloud Provider

`tensorcircuit.cloud` exposes quantum hardware through one set of APIs (`submit_task`, `get_task`, `Task.results`, ...) in `apis.py`. Each provider is a module implementing a fixed set of functions, and `apis.py` dispatches to it by name. Anything specific to a provider belongs in its module, rather than in branches on provider names in the generic layer. Files below are in `tensorcircuit/cloud/` unless a path is given.

Existing providers are good references: `tencent.py` (batch submission, qubit remapping by the provider), `tianyan.py` (vendor SDK), and `lqcloud.py` (vendor SDK, local compilation to native gates, the request body as sent kept in `task.prepared`).

## Registration

1. Add the module, e.g. `myprovider.py`, and register it in `_provider_modules` in `apis.py` together with the vendor SDK modules it needs, whose versions the task store records. Modules are imported on first use, so a missing vendor SDK only fails when the provider is used.
2. Import the vendor SDK inside a helper, raising an `ImportError` with the install command, since vendor SDKs are not dependencies of TensorCircuit. Pin the tested version in `requirements/requirements-extra.txt` for CI, without adding a `pyproject.toml` extra.
3. Put the default service URL in `config.py`.
4. Regenerate the API docs with `docs/source/generate_rst.py` instead of editing `.rst` files, and add a CHANGELOG entry.

## Interface

Functions are looked up by name; a missing one makes the corresponding API raise `NotImplementedError`.

| Function | Contract |
| --- | --- |
| `list_devices(token, **kws)` | `List[Device]`, named `"provider::device"` |
| `list_properties(device, token)` | device properties, see below |
| `submit_task(device, token, **task_kws)` | a `Task`, or a list of `Task` for batches |
| `get_task_details(task, device, token, prettify)` | the task details, see below |
| `resubmit_task(task, token)` | rerun what was prepared for the task as a new `Task` |
| `remove_task(task, token)` | cancel or delete the task |
| `list_tasks(device, token, **filter_kws)` | `List[Task]` |
| `prepare_task(device, token, **task_kws)` (optional) | compile locally before sending, see below |
| `task_metadata(details, prepared)` (optional) | the metadata added by TensorCircuit, see below |

### Properties

Besides the vendor fields, `list_properties` should provide the standardized ones that `Device.topology()`, `Device.native_gates()` and the compilers read:

- `bits`: `{qubit: {"T1", "T2", "ReadoutF0Err", "ReadoutF1Err", "SingleQubitErrRate", ...}}`, None for unknown values;
- `links`: `{(q1, q2): {"CZErrRate", ...}}`, with both directions for symmetric couplers;
- `native_gates`: qiskit gate names.

When the provider compiles locally, compile against these standardized properties rather than the raw vendor configuration, so that what users see is what circuits are compiled to. Facts the vendor does not report, e.g. the native gates chosen by the provider, are added here once.

The error rates `SingleQubitErrRate`, `ReadoutF0Err`, `ReadoutF1Err` and `CZErrRate` drive the noise aware layout of `compiler.qiskit_compiler.properties_to_target`; a qubit or coupler with any of them unknown is not used, so report what the vendor gives rather than defaults.

### Submission

`apis.submit_task` records a submission intent in the task store, then calls the provider `submit_task` once. Paid requests must never be retried blindly, since a retry after a lost response may run and charge the tasks twice. The provider must therefore classify every failure:

- **Local failures** before anything is sent, e.g. an unsupported circuit or invalid shots: raise them as they are, typically `ValueError`. The intent is deleted.
- **Explicit rejection** by the server, e.g. HTTP 4xx such as invalid credentials or circuits: raise `utils.SubmissionRejected`. The intent is deleted.
- **Anything after which the tasks may have been accepted**, e.g. timeouts, connection errors, HTTP 5xx and 408, idempotency conflicts, malformed responses, or a response without a task id: raise `utils.SubmissionUnknown`. The intent is kept as `unknown`, to be bound later by `TaskStore.bind(local_id, remote_id)`.
- **After the server accepts the tasks, never raise**: the paid task id would be lost. Return the tasks even if some optional bookkeeping fails.

To keep the classification sharp, fetch everything the submission needs, e.g. the device configuration, before sending, so that the send itself is the only request that can fail ambiguously. If the vendor SDK retries internally, make sure it uses an idempotency key that the server deduplicates.

For batches, return the accepted tasks with `task.circuit_index` set to their circuit in the batch; a circuit without a returned task counts as explicitly rejected. A response that does not account for every circuit is `SubmissionUnknown`.

Ignore the options of other providers, since generic callers pass one set of options to every provider. `wrapper.batch_expectation_ps` submits circuits already compiled onto the physical qubits with `compiling=False`, `enable_qos_qubit_mapping=False` and `enable_qos_gate_decomposition=False`; with these, qubit `i` of a circuit must run on physical qubit `i`, whose readout errors are taken from `bits` for the mitigation.

### What is prepared: `task.prepared`

`task.prepared` is a dict of what TensorCircuit prepared for sending one circuit, which is not necessarily the request the server receives. It is stored in the task record and used by `resubmit_task`:

- the prepared program, e.g. `source` for `tencent` and `compilation` for `lqcloud`;
- `positional_logical_mapping`: from the classical bit, i.e. the position in the result bitstring, to the qubit it measures in the prepared circuit; None if unknown, e.g. for raw source strings. It is derived from the circuit and never sent, since reading it back from a serialized program would need a parser per format;
- optionally the request actually sent, if the provider can capture it after sending, e.g. `payload` for `lqcloud`.

Without `prepare_task`, it holds the mapping of each circuit's `get_positional_logical_mapping()`. Implement `prepare_task(device, token, **task_kws) -> (send_kws, prepared)` when the provider compiles locally: it returns the keyword arguments for `submit_task`, e.g. the compiled program, and one `prepared` dict per circuit, so that the intent recorded before sending already holds the exact program. `submit_task` may also set `task.prepared` itself, e.g. to add the request body captured after sending; it then replaces the dict from `prepare_task`, so it must be the full dict.

Keep every option needed to send the program again in the prepared program, e.g. the shots and the SDK run options in the `lqcloud` compilation, since `submit_task` and `resubmit_task` only get the prepared program. Pass vendor run options as one dict, e.g. `run_options` of `lqcloud`, rather than one argument each, and allow only the options known to be read, if the SDK silently ignores unknown ones.

Record facts, not predictions. If an SDK rewrites the program before sending, e.g. by adding resets or readout pulses, capture what it actually sends, e.g. with a `requests` response hook on the session of the provider instance created for this submission, instead of re-implementing or calling its private logic. Where the compilation itself depends on such a decision, e.g. the two-state readout of `lqcloud` deciding the `x21` restores after mid-circuit measurements, call the SDK rule rather than re-implementing it, record the decision in the prepared program, and check it again when sending.

### Task details

`get_task_details` returns the vendor fields plus:

- `state`: one of `"pending"`, `"completed"` and `"failed"`. Map every vendor state explicitly, and raise `ValueError` for unknown ones instead of guessing. `Task.results` waits on `pending`, so do not add timeouts there.
- `results`: the observed counts without any correction once completed. Bit `i` of each bitstring, from the left, is classical bit `i`, i.e. the measurement with `record_index` `i`. Only measured classical bits appear. Counts corrected by the vendor, if any, go into the vendor fields.
- `memory`: the bitstring of each shot, if the vendor returns it.
- `err`: the error message when failed, raised by `Task.results` as `TaskFailed`.

Reject result formats you cannot read, e.g. results offloaded to a URL, instead of returning partial data. Handle every format the vendor may choose by itself; for example, LQCloud returns per-shot `memory` instead of counts for 12 or more measured qubits.

### Metadata: `task_metadata(details, prepared)`

`Task.details()["tensorcircuit"]` holds what TensorCircuit adds to the provider details and what the provider cannot give alone, with the same meaning on every provider, i.e. the keys in `metadata_keys` of `abstraction.py`. `task_metadata` returns the values the provider knows from its details and `task.prepared`, empty if unknown; the other values are None, i.e. unknown:

- `positional_logical_mapping`: from the classical bit to the physical qubit. Return the prepared mapping if the provider runs the prepared qubits unchanged, see `lqcloud.py`, or compose it with the provider's qubit mapping if the provider remaps them, see `tencent.py`.

Add a key to `metadata_keys` only when it means the same thing on different providers and needs TensorCircuit's side, e.g. the readout windows of mid-circuit measurements; what the provider observes belongs in the task details above, and vendor-only data, e.g. IQ points or readout-corrected counts, stays in the provider's own fields.

## Circuits and compilation

- Raise on what the hardware cannot run, e.g. mid-circuit measurements or resets on hardware without them, noise channels, MPS or density-matrix inputs, or batch submissions, instead of dropping it silently. A state vector `inputs` can be run by synthesizing a state preparation circuit, e.g. by `compiler.qiskit_compile`, whose depth grows exponentially with the number of qubits; document this cost.
- With mid-circuit measurements, each measurement writes its own classical bit, i.e. its `record_index`, and the measured qubit goes on in the measured state unless a `reset_instruction` follows; reject circuits writing a classical bit twice, and add what the hardware needs to keep this meaning, e.g. `x21` after the two-state readout of `lqcloud`.
- When compiling locally with `compiler.qiskit_compile`, separate each readout window, i.e. the final measurements and each run of mid-circuit measurements, from the gates by barriers (`remove_barriers=False`) so the optimizer cannot move the measurements, and pass a fixed `seed_transpiler` so recompiling gives the same program.
- When renaming gates to the vendor's native gates, check their conventions. With only `rz`, `sx`, `x`, `cz`, |0> inputs and Z-basis readout, flipping the rotation sense of `rz` or `sx` is unobservable, since it amounts to the complex conjugation of the circuit, and any basis change is built from the same gates. The angle scale, e.g. `rz(theta)` rotating by `theta / 2`, is observable and must be confirmed on hardware with circuits of arbitrary angles. A second non-diagonal native gate, or gates decomposed by the vendor, make the sign observable as well.

## Credentials

Tokens come only from TensorCircuit: `set_token(token, provider=...)`, the `TC_TOKEN_<PROVIDER>` environment variable, or the auth file, passed in as `token`. Do not let the vendor SDK read its own environment variables or account files, and turn off any local job store of the SDK, since the TensorCircuit task store already records the tasks. Never write tokens into logs, task records or tests.

## Tests

Put the tests in `tests/test_<provider>.py`:

- Offline tests run by default. Run the real SDK against a fake server, patched at the lowest HTTP level used, e.g. `requests.adapters.HTTPAdapter.send`, so that the SDK's request building, compression and hooks are exercised. Make the fake fail on unexpected requests. Set decoy SDK credentials in the environment to check that they are ignored.
- Cover the submission error classification, the state mapping including unknown states, result formats, bit order with partial measurements, the store records and `resubmit`.
- Online tests need `TC_CLOUD_TEST=1` and `TC_TOKEN_<PROVIDER>`; paid hardware tests also need `TC_CLOUD_HARDWARE_TEST=1`. Read the switches at import time so that tokens set by fixtures cannot enable them.

On hardware, check what the offline tests cannot:

- the bit order, with a partial measurement out of qubit order, e.g. `x(2)` and measuring qubits 2, 0, 1 should give mostly `100`;
- the qubit layout, routing and gate conventions against the ideal distribution, with circuits of arbitrary rotation angles, using statistical checks such as the set of leading outcomes or a total variation distance bound, not exact rankings;
- every result format the vendor may return.

Keep hardware circuits shallow: deep state preparation fails from decoherence alone and hides integration errors. Keep a log of every paid run with its task id, purpose, result and cost.
