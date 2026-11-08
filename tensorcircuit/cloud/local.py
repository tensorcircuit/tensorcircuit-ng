"""
Cloud provider from local machine
"""

from typing import Any, Dict, List, Optional, Union, Sequence
from uuid import uuid4
import time

from .abstraction import Device, sep, Task
from ..cons import backend
from ..quantum import sample2all
from ..utils import is_sequence
from ..abstractcircuit import AbstractCircuit

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
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    **kws: Any,
) -> List[Task]:
    if kws.get("source") is not None:
        raise ValueError("Local provider only runs circuits, `source` is unsupported")

    def _circuit2result(c: AbstractCircuit) -> Dict[str, Any]:
        if device.name not in ["testing", "default"]:
            raise ValueError("Unsupported device from local provider: %s" % device.name)
        if any(
            d["name"] == "measure"
            and any(d["index"][0] in g["index"] for g in c._qir[d.get("pos", 0) :])
            for d in c._extra_qir
        ):
            raise ValueError("Local provider cannot run gates after a measurement")
        samples = c.sample(batch=shots, allow_state=True, format="sample_bin")  # type: ignore
        # bit i of the result is the qubit measured into classical bit i, unwritten
        # classical bits read the appended zero column, and all qubits are measured
        # without measure instructions
        positions = c.get_positional_logical_mapping()
        columns = [positions.get(i, c._nqubits) for i in range(max(positions) + 1)]
        samples = backend.concat([samples, backend.zeros_like(samples[:, :1])], axis=1)
        samples = backend.gather(samples, backend.convert_to_tensor(columns), axis=1)
        count = sample2all(samples, len(columns), format="count_dict_bin")
        d = {
            "id": str(uuid4()),
            "state": "completed",
            "at": time.time() * 1e6,
            "shots": shots,
            "device": device.name,
            "results": count,
        }
        return d

    if is_sequence(circuit):
        tl = []
        for c in circuit:  # type: ignore
            d = _circuit2result(c)
            task_list[d["id"]] = d
            tl.append(Task(id_=d["id"], device=device))
        return tl
    else:
        d = _circuit2result(circuit)  # type: ignore
        task_list[d["id"]] = d

        return Task(id_=d["id"], device=device)  # type: ignore


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    r = []
    for t, v in task_list.items():
        if (device is not None and v["device"] == device.name) or device is None:
            r.append(Task(id_=t, device=Device.from_name("local" + sep + v["device"])))
    return r
