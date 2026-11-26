"""
Cloud provider from Qboson (玻色量子), the coherent Ising machine (CIM)
https://platform.qboson.com

The CIM minimizes the energy of spins :math:`s_i \\in \\{+1, -1\\}`, given by
a coupling matrix :math:`J` and optional fields :math:`h`:

.. math::

    E(s) = -s^T J s - h^T s

and returns a few low energy spin configurations. Only the optimization mode
of the Kaiwu SDK is supported.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import io
import os
import pickle
import tempfile
import uuid
import zlib

import numpy as np
import requests

from .abstraction import Device, sep, Task
from .utils import SubmissionRejected, SubmissionUnknown
from ..utils import is_sequence

devices = ["cim"]
# the size of the coupling matrix, including the auxiliary spin for fields
max_spins = 1000
# Kaiwu task states, i.e. ``data.code`` of the task query, to the unified ones
_states = {
    "2001": "pending",  # validating
    "2002": "pending",  # calculating
    "1000": "completed",
    "4001": "failed",  # validation failed
}
_task_not_found = "10320"
_result_format = "ndarray-zlib-v1"
# the SDK requires a checkpoint directory, while the server finds a task by its
# name and matrix, so the checkpoints are only a local cache
_checkpoint_dir = os.path.join(tempfile.gettempdir(), "tc-qboson")
_licensed: Optional[str] = None

Tensor = Any
IsingSource = Union[Any, Dict[str, Any]]


def _kaiwu() -> Any:
    try:
        import kaiwu
    except ImportError as e:
        raise ImportError(
            "kaiwu is required for the Qboson provider, which needs Python 3.10, "
            '3.12 or 3.14. Install it via: pip install "kaiwu>=1.4.4,<1.5"'
        ) from e
    return kaiwu


def _optimizer(token: Optional[str], task_name: str) -> Any:
    """
    The Kaiwu optimizer of one task. The license of the token is written by the
    SDK into its package directory, shared by every thread and process of the
    Python environment, so one environment uses one account at a time.
    """
    global _licensed
    if not token or ":" not in token:
        raise ValueError(
            "Qboson provider requires the token as `<user id>:<SDK code>`, set it via "
            '`tc.cloud.apis.set_token("<user id>:<SDK code>", provider="qboson")`'
        )
    kw = _kaiwu()
    if _licensed != token:
        user_id, sdk_code = token.split(":", 1)
        kw.license.init(user_id=user_id.strip(), sdk_code=sdk_code.strip())
        _licensed = token
    manager = kw.common.CheckpointManager
    saved = manager.save_dir
    os.makedirs(_checkpoint_dir, exist_ok=True)
    manager.save_dir = _checkpoint_dir
    try:
        return kw.cim.CIMOptimizer(
            task_name=task_name, wait=False, task_mode=kw.cim.TaskMode.OPTIMIZATION
        )
    finally:
        # the directory is a class attribute, shared with the user's own SDK use
        manager.save_dir = saved


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    """
    List the Qboson devices, i.e. ``cim``, since the SDK cannot list them.

    :param token: unused, for the uniform provider interface
    :type token: Optional[str], optional
    :param kws: unused, for the uniform provider interface
    :type kws: Any
    :return: the devices
    :rtype: List[Device]
    """
    return [Device.from_name("qboson" + sep + d) for d in devices]


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    """
    The properties of the CIM, which the SDK does not report.

    :param device: the device
    :type device: Device
    :param token: unused, for the uniform provider interface
    :type token: Optional[str], optional
    :return: ``max_spins``, the size of the coupling matrix including the
        auxiliary spin for fields, and ``modes``, the supported task modes
    :rtype: Dict[str, Any]
    """
    _check_device(device)
    return {"id": device.name, "max_spins": max_spins, "modes": ["optimization"]}


def _check_device(device: Device) -> None:
    if device.name not in devices:
        raise ValueError(
            "Unknown Qboson device %s, available: %s" % (device.name, devices)
        )


def _matrix(j: Any) -> Tensor:
    if hasattr(j, "toarray"):
        j = j.toarray()
    if not isinstance(j, np.ndarray):
        raise ValueError(
            "`J` must be a numpy array or a scipy sparse matrix, got %s; a list or "
            "tuple is a batch of problems" % type(j).__name__
        )
    j = np.asarray(j, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != j.shape[1] or j.shape[0] == 0:
        raise ValueError("`J` must be a nonempty square matrix, got %s" % (j.shape,))
    if not np.all(np.isfinite(j)):
        raise ValueError("`J` must be finite")
    if not np.array_equal(j, j.T):
        raise ValueError("`J` must be symmetric")
    if np.any(np.diag(j) != 0):
        raise ValueError("`J` must have a zero diagonal, which only shifts the energy")
    return j


def ising_matrix(source: IsingSource) -> Tuple[Tensor, bool]:
    """
    The coupling matrix sent to the CIM for an Ising problem. Fields are
    absorbed by an auxiliary last spin :math:`s_a`, coupled to spin :math:`i`
    by :math:`h_i / 2`, so that :math:`E = -s^T J s - s_a h^T s`, whose
    spins of the problem are :math:`s_i s_a` whichever sign :math:`s_a` takes.

    :param source: the coupling matrix ``J``, or ``{"J": J, "h": h}`` with the
        fields ``h``
    :type source: Union[Any, Dict[str, Any]]
    :return: the coupling matrix as a C contiguous float64 array without
        negative zeros, since the SDK identifies a task by the hash of its
        serialization, and whether the auxiliary spin is added
    :rtype: Tuple[Tensor, bool]
    """
    if not isinstance(source, dict):
        return np.ascontiguousarray(_matrix(source) + 0.0), False
    if set(source) != {"J", "h"}:
        raise ValueError(
            'An Ising problem with fields is given as `{"J": J, "h": h}`, got '
            "the keys %s" % sorted(source)
        )
    j = _matrix(source["J"])
    n = j.shape[0]
    h = np.asarray(source["h"], dtype=np.float64)
    if h.shape != (n,) or not np.all(np.isfinite(h)):
        raise ValueError("`h` must be a finite vector of length %d" % n)
    m = np.zeros([n + 1, n + 1])
    m[:n, :n] = j
    m[:n, n] = m[n, :n] = h / 2
    return m + 0.0, True


def _new_name() -> str:
    return "tc-%s" % uuid.uuid4().hex[:16]


def prepare_task(
    device: Device,
    token: Optional[str],
    source: Optional[Union[IsingSource, List[IsingSource]]] = None,
    circuit: Any = None,
    shots: Any = None,
    **kws: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Prepare the coupling matrices and task names sent by :py:func:`submit_task`.

    :param device: the device
    :type device: Device
    :param token: unused, for the uniform provider interface
    :type token: Optional[str]
    :param source: the Ising problem, or a list for a batch, see :py:func:`submit_task`
    :type source: Optional[Union[IsingSource, List[IsingSource]]], optional
    :param circuit: unsupported, raises if given, defaults to None
    :type circuit: Any, optional
    :param shots: unsupported, raises if given, defaults to None
    :type shots: Any, optional
    :param kws: options of other providers, ignored
    :type kws: Any
    :return: the keyword arguments of :py:func:`submit_task` with the
        ``problem`` to send, a list for a batch, and what is prepared for each
        problem, i.e. its ``problem``
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]
    """
    if circuit is not None:
        raise ValueError(
            "Qboson provider runs Ising problems, give the coupling matrix as `source`"
        )
    if shots is not None:
        raise ValueError("Qboson provider supports only the optimization mode")
    if source is None:
        raise ValueError("`source` must be provided")
    _check_device(device)
    batch = is_sequence(source)
    problems = []
    for s in source if batch else [source]:
        # checked before a sparse matrix is made dense
        aux = isinstance(s, dict)
        shape = np.shape(s.get("J") if aux else s)  # type: ignore
        if shape and shape[0] + aux > max_spins:
            raise ValueError(
                "%d spins%s exceed the %d of the device"
                % (shape[0] + aux, " with the auxiliary spin" if aux else "", max_spins)
            )
        m, _ = ising_matrix(s)
        problems.append({"matrix": m, "auxiliary": aux, "task_name": _new_name()})
    send = problems if batch else problems[0]
    return {"problem": send}, [{"problem": p} for p in problems]


def _send(optimizer: Any, matrix: Tensor) -> None:
    """
    Create the task once, where the SDK first looks up the task of the same
    name and matrix. The SDK raises ``ValueError("Task creation failed: ...")``
    when the server rejects the task, and wraps the failures of the requests in
    ``ValueError`` with the ``requests`` exception as the cause, which may come
    after the task is created.
    """
    try:
        optimizer.solve(matrix)
    except ValueError as e:
        if isinstance(e.__cause__, requests.RequestException):
            raise SubmissionUnknown(str(e)) from e
        if str(e).startswith("Task creation failed"):
            raise SubmissionRejected(str(e)) from e
        raise
    except requests.RequestException as e:
        raise SubmissionUnknown(repr(e)) from e


def submit_task(
    device: Device,
    token: str,
    source: Optional[Union[IsingSource, List[IsingSource]]] = None,
    problem: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    **kws: Any,
) -> Union[Task, List[Task]]:
    """
    Submit Ising problems to the Qboson CIM in the optimization mode, which
    returns a few low energy spin configurations, typically ten but possibly
    fewer, as the counts of their bitstrings.

    The variables are spins :math:`s_i \\in \\{+1, -1\\}`, not binary ones.
    ``source`` is a symmetric matrix ``J`` with a zero diagonal, as a numpy
    array or a scipy sparse matrix, for the energy :math:`-s^T J s`, or a dict
    ``{"J": J, "h": h}`` with the fields ``h`` for :math:`-s^T J s - h^T s`.
    Bit ``i`` of the results is ``"0"`` for :math:`s_i = +1` and ``"1"`` for
    :math:`s_i = -1`, so a binary variable :math:`x = (1 - s) / 2` is the bit
    itself: for a QUBO problem converted to ``J`` and ``h`` by this relation,
    the bitstrings are its solutions. Do not pass a QUBO matrix as ``J``.

    Fields are absorbed by an auxiliary spin, see :py:func:`ising_matrix`, which
    takes one spin of the device and is removed from the results. A matrix with
    an auxiliary spin added by the user, e.g. by
    ``kaiwu.core.qubo_matrix_to_ising_matrix``, runs as it is, and its results
    must be decoded by the user.

    The CIM quantizes the matrix itself, e.g. to 8-bit integers, so a matrix of
    a large dynamic range may lose precision; evaluate the energies of the
    results on the original problem.

    A list or tuple of problems is a batch, so a nested list is not a matrix.
    Each
    problem is a separate task, submitted one after another; if one fails, the
    earlier ones are already submitted and ``SubmissionUnknown`` names them.
    The matrix and the task name are kept in ``task.prepared["problem"]``, also
    in the task store, since the server finds a task by both.

    :param device: the device, ``cim``
    :type device: Device
    :param token: ``"<user id>:<SDK code>"`` of the Qboson platform; the SDK
        keeps the license in its package directory, so one Python environment
        uses one account at a time, also across threads and processes
    :type token: str
    :param source: the Ising problem ``J`` or ``{"J": J, "h": h}``, or a list
        for a batch
    :type source: Optional[Union[IsingSource, List[IsingSource]]], optional
    :param problem: a prepared problem to send directly, or a list for a batch,
        e.g. by :py:func:`prepare_task`, defaults to None
    :type problem: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
    :param kws: options of other providers, ignored, except ``circuit`` and
        ``shots``, which raise
    :type kws: Any
    :return: the task, whose id is its task name, or a list of tasks for a batch
    :rtype: Union[Task, List[Task]]
    """
    if problem is None:
        send_kws, _ = prepare_task(device, token, source, **kws)
        problem = send_kws["problem"]
    batch = is_sequence(problem)
    tasks: List[Task] = []
    for i, p in enumerate(problem if batch else [problem]):  # type: ignore
        try:
            _send(_optimizer(token, p["task_name"]), p["matrix"])
        except (SubmissionRejected, SubmissionUnknown) as e:
            if not tasks:
                raise
            # the earlier tasks of the batch are accepted, and found by their names
            raise SubmissionUnknown(
                "Qboson problem %d of the batch failed after tasks %s were "
                "submitted: %s" % (i, [t.id_ for t in tasks], e)
            ) from e
        task = Task(p["task_name"], device=device)
        task.circuit_index = i
        task.prepared = {"problem": p}
        tasks.append(task)
    return tasks if batch else tasks[0]


def resubmit_task(task: Task, token: str) -> Task:
    """Raise because a rerun is a new submission of the same problem."""
    raise NotImplementedError(
        "Qboson provider does not support resubmit_task, submit the problem "
        "again by `submit_task` instead"
    )


def remove_task(task: Task, token: str) -> Any:
    """Raise because the Kaiwu SDK has no cancellation."""
    raise NotImplementedError(
        "Qboson provider does not support remove_task because the Kaiwu SDK "
        "has no cancellation"
    )


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    """Raise because the Kaiwu SDK has no task listing."""
    raise NotImplementedError(
        "Qboson provider does not support list_tasks because the Kaiwu SDK has "
        "no task listing, find the tasks in the task store instead"
    )


def _problem(task: Task) -> Dict[str, Any]:
    p = (task.prepared or {}).get("problem")
    if p is None:
        raise ValueError(
            "The problem of Qboson task %s is unknown, while the server finds a "
            "task by its matrix; keep the Task object, or record the tasks by "
            "`tc.cloud.apis.set_store(True)`" % task.id_
        )
    # the matrix restored from the task store is a nested list
    return dict(p, matrix=np.ascontiguousarray(p["matrix"], dtype=np.float64))


class _ResultUnpickler(pickle.Unpickler):
    # the result holds only builtin values, so no class is ever loaded
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            "Unexpected %s.%s in a Qboson result" % (module, name)
        )


def _solutions(url: str) -> Tensor:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    d = _ResultUnpickler(io.BytesIO(r.content)).load()
    if not isinstance(d, dict) or d.get("format") != _result_format:
        raise ValueError("Unsupported Qboson result format: %r" % (d,))
    data = zlib.decompress(d["data"])
    return np.frombuffer(data, dtype=d["dtype"]).reshape(d["shape"])


def _bitstrings(spins: Tensor, auxiliary: bool) -> List[str]:
    if spins.ndim != 2 or not np.all(np.abs(spins) == 1):
        raise ValueError("Unexpected Qboson solutions of shape %s" % (spins.shape,))
    if auxiliary:
        spins = spins[:, :-1] * spins[:, -1:]
    return ["".join("0" if v > 0 else "1" for v in s) for s in spins]


def get_task_details(
    task: Task, device: Device, token: Optional[str], prettify: bool
) -> Dict[str, Any]:
    """
    Get the task details by a single query, with the unified ``state``
    (``pending``, ``completed`` or ``failed``), the counts of the returned
    bitstrings as ``results``, i.e. one count per solution, the bitstrings in
    the order returned by the CIM, i.e. from the lowest energy, as ``memory``,
    and the fields of the query in ``qboson``.

    :param task: the task, with its problem in ``task.prepared``
    :type task: Task
    :param device: the device of the task
    :type device: Device
    :param token: ``"<user id>:<SDK code>"`` of the Qboson platform
    :type token: Optional[str]
    :param prettify: unused, the details hold no python objects to prettify
    :type prettify: bool
    :return: the task details
    :rtype: Dict[str, Any]
    """
    p = _problem(task)
    reply = _optimizer(token, p["task_name"]).get_task_result(p["matrix"])
    if reply.get("code") == _task_not_found:
        raise ValueError("Qboson task %s is not found on the server" % task.id_)
    data = reply.get("data") or {}
    code = data.get("code")
    if reply.get("code") != "0" or code not in _states:
        raise ValueError("Unknown Qboson task status of %s: %r" % (task.id_, reply))
    details: Dict[str, Any] = {
        "id": task.id_,
        "state": _states[code],
        "device": str(device),
        "results": {},
        "shots": None,
        "qboson": {k: v for k, v in data.items() if k != "result"},
    }
    if details["state"] == "failed":
        details["err"] = data.get("failed_desc") or data.get("task_status") or code
    elif details["state"] == "completed":
        memory = _bitstrings(_solutions(data["result"]), p["auxiliary"])
        counts: Dict[str, int] = {}
        for b in memory:
            counts[b] = counts.get(b, 0) + 1
        details["results"] = counts
        details["memory"] = memory
        details["shots"] = len(memory)
    return details
