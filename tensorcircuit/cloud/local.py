"""
Cloud provider from local machine
"""

from typing import Any, Dict, List, Optional, Union, Sequence
from uuid import uuid4
import time

from .abstraction import Device, sep, Task
from ..circuit import Circuit
from ..cons import backend
from ..quantum import sample2all

local_devices = ["testing"]

task_list: Dict[str, Any] = {}  # memory only task cache


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    rs = []
    for d in local_devices:
        rs.append(Device.from_name("local" + sep + d))
    return rs


# the prepared circuit runs on its qubits as they are
def task_metadata(details: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return {"positional_logical_mapping": prepared.get("positional_logical_mapping")}


def get_task_details(
    task: Task, device: Device, token: str, prettify: bool
) -> Dict[str, Any]:
    if task.id_ in task_list:
        return task_list[task.id_]  # type: ignore
    raise ValueError("no task with id: %s" % task.id_)


def submit_task(
    device: Device,
    token: str,
    shots: Union[int, Sequence[int]] = 1024,
    version: str = "1",
    circuit: Optional[Union[Circuit, Sequence[Circuit]]] = None,
    **kws: Any,
) -> Union[Task, List[Task]]:
    if circuit is None or kws.get("source") is not None:
        raise ValueError("Local provider only runs circuits, `source` is unsupported")

    if device.name not in ["testing", "default"]:
        raise ValueError("Unsupported device from local provider: %s" % device.name)

    circuits = circuit if isinstance(circuit, Sequence) else [circuit]
    shot_counts = shots if isinstance(shots, Sequence) else [shots] * len(circuits)
    if len(shot_counts) != len(circuits):
        raise ValueError("Expected one shot count per circuit")
    results = {}
    for c, nshots in zip(circuits, shot_counts):
        if not isinstance(c, Circuit) or c._d != 2:
            raise ValueError("Local provider expects a qubit tc.Circuit")
        count = _sample_counts(c, nshots)
        task_id = str(uuid4())
        results[task_id] = {
            "id": task_id,
            "state": "completed",
            "at": time.time() * 1e6,
            "shots": nshots,
            "device": device.name,
            "results": count,
        }
    task_list.update(results)
    tasks = [Task(id_=task_id, device=device) for task_id in results]
    return tasks if isinstance(circuit, Sequence) else tasks[0]


def _sample_counts(c: Circuit, shots: int) -> Dict[str, int]:
    measures = [
        d
        for d in c._merge_qir_with_extra()
        if d.get("instruction") and c._is_measure_instruction_name(d["name"])
    ]
    if not measures:
        return c.sample(  # type: ignore[no-any-return]
            batch=shots, allow_state=True, format="count_dict_bin", jittable=False
        )

    samples = backend.cast(c.sample_measurements(shots=shots), "int32")
    last_write = {d.get("record_index", i): i for i, d in enumerate(measures)}
    width = max(last_write) + 1
    order = [last_write.get(i, len(measures)) for i in range(width)]
    if len(last_write) < width:
        samples = backend.concat([samples, backend.zeros_like(samples[:, :1])], axis=1)
    samples = backend.gather(samples, backend.convert_to_tensor(order), axis=1)
    return sample2all(samples, width, format="count_dict_bin")  # type: ignore[no-any-return]


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    r = []
    for t, v in task_list.items():
        if (device is not None and v["device"] == device.name) or device is None:
            r.append(Task(id_=t, device=Device.from_name("local" + sep + v["device"])))
    return r
