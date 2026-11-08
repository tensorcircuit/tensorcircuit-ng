"""
Local task store for cloud submissions: a SQLite index with gzip JSON records
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from datetime import datetime
import gzip
import hashlib
import importlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from uuid import uuid4

import numpy as np

from ..cons import backend

logger = logging.getLogger(__name__)

schema_version = 1
default_store_path = os.path.join(os.path.expanduser("~"), ".tensorcircuit", "tasks")

# failures of the store must never break the task submission or query itself
store_errors = (OSError, sqlite3.Error, ValueError, TypeError)

_columns = [
    "local_id",
    "provider",
    "device",
    "remote_id",
    "state",
    "submission_status",
    "batch_id",
    "circuit_index",
    "parent_id",
    "created_at",
    "updated_at",
    "body_status",
    "body_sha256",
    "schema_version",
]

_schema = """
CREATE TABLE IF NOT EXISTS tasks (
    local_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    device TEXT NOT NULL,
    remote_id TEXT,
    state TEXT,
    submission_status TEXT NOT NULL,
    batch_id TEXT,
    circuit_index INTEGER,
    parent_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    body_status TEXT NOT NULL,
    body_sha256 TEXT,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_remote_id ON tasks (remote_id);
CREATE INDEX IF NOT EXISTS tasks_created_at ON tasks (created_at);
"""

PathLike = Union[str, "os.PathLike[str]"]


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.generic, np.ndarray)):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, datetime):
        return o.isoformat()
    if backend.is_tensor(o):
        return np.asarray(backend.numpy(o)).tolist()
    return repr(o)


def _dumps(o: Any) -> str:
    return json.dumps(o, default=_jsonable)


def _timestamp(t: Union[float, str, datetime]) -> float:
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    if isinstance(t, datetime):
        return t.timestamp()
    return float(t)


def _versions(sdks: Sequence[str]) -> Dict[str, Optional[str]]:
    from .. import __version__

    versions: Dict[str, Optional[str]] = {"tensorcircuit": __version__}
    for name in sdks:
        # providers import their SDK lazily, possibly only after the intent is made
        try:
            importlib.import_module(name)
        except ImportError:
            pass  # reported by the provider
    # qiskit compiles circuits locally for any provider
    for name in ["qiskit", *sdks]:
        if name in sys.modules:
            versions[name] = getattr(sys.modules[name], "__version__", None)
    return versions


def _int_keys(d: Dict[str, Any]) -> Dict[int, Any]:
    # JSON object keys are strings, while mappings use integer indices
    return {int(i): j for i, j in d.items()}


def circuit_record(c: Any) -> Dict[str, Any]:
    """
    JSON-compatible record of a TensorCircuit circuit in the :py:meth:`to_json`
    format, which can be restored by ``Circuit.from_json``

    :param c: the circuit
    :type c: AbstractCircuit
    :return: the circuit record
    :rtype: Dict[str, Any]
    """
    params = getattr(c, "circuit_param", {})
    return dict(
        json.loads(c.to_json()),
        type=type(c).__name__,
        custom_inputs=any(
            params.get(k) is not None for k in ["inputs", "mps_inputs", "tensors"]
        ),
    )


class TaskStore:
    """
    Local record of cloud tasks in a directory with a SQLite index ``tasks.sqlite``
    and one gzip JSON record ``records/<local_id>.json.gz`` per submitted circuit.
    Records are never deleted automatically, except the submission intents of
    circuits certainly not submitted.

    :Example:

    >>> tc.cloud.apis.set_store(True)  # record every submission from now on
    >>> t = tc.cloud.apis.submit_task(device="local::testing", circuit=c)
    >>> s = tc.cloud.store.TaskStore()
    >>> s.get(t.local_id)["details"]["results"]
    >>> s.find(provider="local", state="completed", limit=10)
    """

    def __init__(self, path: Optional[PathLike] = None):
        """
        :param path: the store directory, defaults to None, i.e. ``~/.tensorcircuit/tasks``
        :type path: Optional[PathLike], optional
        """
        if path is None:
            path = default_store_path
        self.path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        self.index_path = os.path.join(self.path, "tasks.sqlite")
        self.records_path = os.path.join(self.path, "records")

    def __repr__(self) -> str:
        return "TaskStore(%r)" % self.path

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(self.path, mode=0o700, exist_ok=True)
        os.makedirs(self.records_path, mode=0o700, exist_ok=True)
        conn = sqlite3.connect(self.index_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript(_schema)
        return conn

    def _query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        if not os.path.exists(self.index_path):
            return []
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def _body_path(self, local_id: str) -> str:
        return os.path.join(self.records_path, local_id + ".json.gz")

    def _write_body(self, record: Dict[str, Any]) -> str:
        data = gzip.compress(_dumps(record).encode("utf-8"))
        fd, tmp = tempfile.mkstemp(
            prefix=".%s." % record["local_id"], suffix=".tmp", dir=self.records_path
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, self._body_path(record["local_id"]))
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _row(record: Dict[str, Any], body_status: str, sha: Optional[str]) -> List[Any]:
        values = dict(record)
        values.update(
            submission_status=record["submission"]["status"],
            body_status=body_status,
            body_sha256=sha,
        )
        return [values[k] for k in _columns]

    def save(self, records: Sequence[Dict[str, Any]]) -> None:
        """
        Insert or overwrite full records, see :py:func:`make_records`.
        The index is marked pending before the record body is replaced and
        complete afterwards, so an interrupted write never looks complete.

        :param records: the records
        :type records: Sequence[Dict[str, Any]]
        """
        sql = "INSERT OR REPLACE INTO tasks (%s) VALUES (%s)" % (
            ", ".join(_columns),
            ", ".join("?" for _ in _columns),
        )
        now = time.time()
        for r in records:
            r["updated_at"] = now
        conn = self._connect()
        try:
            with conn:
                conn.executemany(sql, [self._row(r, "pending", None) for r in records])
            shas = [self._write_body(r) for r in records]
            with conn:
                conn.executemany(
                    sql,
                    [self._row(r, "complete", s) for r, s in zip(records, shas)],
                )
        finally:
            conn.close()

    def update(self, local_id: str, **fields: Any) -> None:
        """
        Update fields of a saved record, e.g. ``state`` and ``details``.

        :param local_id: local id of the record
        :type local_id: str
        :param fields: the fields to update
        :type fields: Any
        """
        rows = self._query("SELECT * FROM tasks WHERE local_id = ?", [local_id])
        if not rows:
            raise ValueError("No task record with local id %s" % local_id)
        record = self._load(rows[0])
        if record["integrity"] not in ["complete", "pending"]:
            raise ValueError("Task record %s is %s" % (local_id, record["integrity"]))
        record.pop("integrity")
        record.update(fields)
        self.save([record])

    def delete(self, local_ids: Sequence[str]) -> None:
        """
        Delete records, e.g. the submission intents of circuits that are
        certainly not submitted.

        :param local_ids: local ids of the records
        :type local_ids: Sequence[str]
        """
        if not local_ids or not os.path.exists(self.index_path):
            return
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    "DELETE FROM tasks WHERE local_id = ?", [[i] for i in local_ids]
                )
        finally:
            conn.close()
        for i in local_ids:
            if os.path.exists(self._body_path(i)):
                os.remove(self._body_path(i))

    def bind(self, local_id: str, remote_id: str) -> Any:
        """
        Bind the remote task id to a record, e.g. a submission of ``unknown``
        status that is found on the provider, and recover the task.

        :param local_id: local id of the record
        :type local_id: str
        :param remote_id: remote task id on the provider
        :type remote_id: str
        :return: the task bound to this store
        :rtype: Task
        """
        self.update(
            local_id,
            remote_id=remote_id,
            submission={"status": "submitted", "error": None},
        )
        return self.task(local_id)

    def _load(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with open(self._body_path(row["local_id"]), "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return dict(row, integrity="missing")
        if row["body_status"] == "complete" and (
            hashlib.sha256(data).hexdigest() != row["body_sha256"]
        ):
            return dict(row, integrity="corrupt")
        try:
            record = json.loads(gzip.decompress(data).decode("utf-8"))
        except (OSError, EOFError, ValueError):
            return dict(row, integrity="corrupt")
        for d in [record.get("prepared") or {}, record.get("tensorcircuit") or {}]:
            if d.get("positional_logical_mapping") is not None:
                d["positional_logical_mapping"] = _int_keys(
                    d["positional_logical_mapping"]
                )
        record["integrity"] = row["body_status"]
        if row["body_status"] == "pending":
            # an interrupted save leaves an older body, while the index holds the
            # latest indexed fields, e.g. the remote id needed for recovery
            record.update({k: row[k] for k in _columns if k in record})
            if record["submission"]["status"] != row["submission_status"]:
                record["submission"] = {
                    "status": row["submission_status"],
                    "error": None,
                }
        return record  # type: ignore

    def get(
        self,
        local_id: Optional[str] = None,
        remote_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the full record by the local id or by the remote task id (the latest one).
        The ``integrity`` field is ``complete``, ``pending`` (an interrupted write),
        ``corrupt`` or ``missing``; for the latter two only the index fields are returned.

        :param local_id: local id of the record, i.e. ``task.local_id``
        :type local_id: Optional[str], optional
        :param remote_id: remote task id on the provider
        :type remote_id: Optional[str], optional
        :param provider: provider name to disambiguate ``remote_id``
        :type provider: Optional[str], optional
        :return: the record or None if not found
        :rtype: Optional[Dict[str, Any]]
        """
        if local_id is not None:
            rows = self._query("SELECT * FROM tasks WHERE local_id = ?", [local_id])
        elif remote_id is not None:
            sql = "SELECT * FROM tasks WHERE remote_id = ?"
            params = [remote_id]
            if provider is not None:
                sql += " AND provider = ?"
                params.append(provider)
            rows = self._query(sql + " ORDER BY created_at DESC LIMIT 1", params)
        else:
            raise ValueError("Either `local_id` or `remote_id` must be provided")
        if not rows:
            return None
        return self._load(rows[0])

    def find(
        self,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        state: Optional[str] = None,
        batch_id: Optional[str] = None,
        submission_status: Optional[str] = None,
        since: Optional[Union[float, str, datetime]] = None,
        until: Optional[Union[float, str, datetime]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search index entries, the latest first.

        :param provider: provider name
        :type provider: Optional[str], optional
        :param device: device name, optionally with the ``provider::`` prefix
        :type device: Optional[str], optional
        :param state: task state, e.g. ``completed``
        :type state: Optional[str], optional
        :param batch_id: circuits submitted in one ``submit_task`` call share the batch id
        :type batch_id: Optional[str], optional
        :param submission_status: ``submitted``, or ``unknown`` when the submission
            may have been accepted without returning the task id, see :py:meth:`bind`
        :type submission_status: Optional[str], optional
        :param since: earliest creation time, as a timestamp, datetime or ISO string
        :type since: Optional[Union[float, str, datetime]], optional
        :param until: latest creation time, as a timestamp, datetime or ISO string
        :type until: Optional[Union[float, str, datetime]], optional
        :param limit: maximal number of entries
        :type limit: Optional[int], optional
        :return: index entries, use :py:meth:`get` for the full records
        :rtype: List[Dict[str, Any]]
        """
        if device is not None and "::" in device:
            provider, device = device.split("::", 1)
        conds: List[str] = []
        params: List[Any] = []
        for k, v in [
            ("provider", provider),
            ("device", device),
            ("state", state),
            ("batch_id", batch_id),
            ("submission_status", submission_status),
        ]:
            if v is not None:
                conds.append("%s = ?" % k)
                params.append(v)
        if since is not None:
            conds.append("created_at >= ?")
            params.append(_timestamp(since))
        if until is not None:
            conds.append("created_at <= ?")
            params.append(_timestamp(until))
        sql = "SELECT * FROM tasks"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at DESC, circuit_index ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._query(sql, params)

    def task(self, local_id: str) -> Any:
        """
        Recover the ``Task`` object of a record to query the provider again.

        :param local_id: local id of the record
        :type local_id: str
        :return: the task bound to this store
        :rtype: Task
        """
        from .abstraction import Device, Task, sep

        record = self.get(local_id)
        if record is None:
            raise ValueError("No task record with local id %s" % local_id)
        if not record.get("remote_id"):
            status = record.get("submission", {}).get("status")
            raise ValueError(
                "Task record %s has no remote task id, submission status: %s"
                % (local_id, status or record.get("submission_status"))
            )
        t = Task(
            record["remote_id"],
            Device.from_name(record["provider"] + sep + record["device"]),
        )
        t.circuit_index = record.get("circuit_index")
        t.prepared = record.get("prepared")
        t._bind_store(self, local_id, state=record.get("state"))
        return t

    def export(self, path: PathLike, **filters: Any) -> int:
        """
        Export full records selected by :py:meth:`find` filters to a self-contained
        gzip JSON lines file, one record per line. Records never contain credentials.

        :param path: output file, e.g. ``tasks.jsonl.gz``
        :type path: PathLike
        :param filters: the same filters as :py:meth:`find`
        :type filters: Any
        :return: number of exported records
        :rtype: int
        """
        rows = self.find(**filters)
        with gzip.open(os.fspath(path), "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(_dumps(self._load(row)) + "\n")
        return len(rows)


def make_records(
    provider: str,
    device: str,
    requests: Sequence[Dict[str, Any]],
    batch: bool,
    sdks: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Build the records of one ``submit_task`` call in the submission intent state,
    one record per circuit.

    :param provider: provider name
    :type provider: str
    :param device: device name
    :type device: str
    :param requests: one request dict per circuit
    :type requests: Sequence[Dict[str, Any]]
    :param batch: whether the circuits are submitted in a batch, sharing a batch id
    :type batch: bool
    :param sdks: the vendor SDK modules of the provider, whose versions are recorded,
        defaults to ()
    :type sdks: Sequence[str], optional
    :return: the records
    :rtype: List[Dict[str, Any]]
    """
    now = time.time()
    batch_id = uuid4().hex if batch else None
    versions = _versions(sdks)
    return [
        {
            "schema_version": schema_version,
            "local_id": uuid4().hex,
            "provider": provider,
            "device": device,
            "remote_id": None,
            "state": None,
            "batch_id": batch_id,
            "circuit_index": i,
            "parent_id": None,
            "created_at": now,
            "updated_at": now,
            "submission": {"status": "intent", "error": None},
            "request": req,
            "prepared": None,
            "details": None,
            "tensorcircuit": None,
            "versions": versions,
        }
        for i, req in enumerate(requests)
    ]


def try_store(f: Callable[..., Any], *args: Any, **kws: Any) -> Optional[str]:
    """
    Call a task store method, where a failure only emits a warning so that it
    never breaks the task submission or query itself.

    :param f: the task store method
    :type f: Callable[..., Any]
    :return: the error message, None on success
    :rtype: Optional[str]
    """
    try:
        f(*args, **kws)
    except store_errors as e:
        logger.warning("task store write failure: %s", e)
        return str(e)
    return None


def as_store(store: Union[bool, PathLike, TaskStore, None]) -> Optional[TaskStore]:
    """
    Resolve the ``store`` argument: ``True`` for the default directory,
    a path for the given directory, ``False`` or ``None`` for no store.
    """
    if store is None or store is False:
        return None
    if store is True:
        return TaskStore()
    if isinstance(store, TaskStore):
        return store
    return TaskStore(store)
