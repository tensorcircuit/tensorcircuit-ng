"""
main entrypoints of cloud module
"""

from typing import Any, Callable, List, Optional, Dict, Sequence, Union, Tuple
from base64 import b64decode, b64encode
from copy import copy
from functools import partial
import importlib
import json
import os
import sys
import tempfile
import logging

from .abstraction import Provider, Device, Task, metadata_keys, sep, sep2
from .store import (
    PathLike,
    TaskStore,
    as_store,
    circuit_record,
    make_records,
    store_errors,
    try_store,
)
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence

try:
    from .utils import unknown_submission_errors
except ImportError:
    # without requests, no remote provider can submit tasks
    unknown_submission_errors = ()  # type: ignore

logger = logging.getLogger(__name__)

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]

# provider name -> the module implementing the provider interface, see
# ``_provider_api``, and the vendor SDKs it needs, recorded in the task store
_provider_modules = {
    "tencent": ("tencent", []),
    "local": ("local", []),
    "quafu": ("quafu_provider", ["quafu"]),
    "tianyan": ("tianyan", ["cqlib"]),
    "lqcloud": ("lqcloud", ["lqcloud"]),
}

default_provider = Provider.from_name("tencent")
avail_providers = list(_provider_modules)


def _provider_module(provider: Provider) -> Any:
    if provider.name not in _provider_modules:
        raise ValueError("Unsupported provider: %s" % provider.name)
    return importlib.import_module(
        "." + _provider_modules[provider.name][0], __package__
    )


def _provider_api(provider: Provider, name: str) -> Callable[..., Any]:
    """
    Get a function of the provider interface implemented by each provider module:
    ``list_devices(token, **kws)``, ``list_properties(device, token)``,
    ``submit_task(device, token, **task_kws)``,
    ``get_task_details(task, device, token, prettify)``,
    ``resubmit_task(task, token)``, ``remove_task(task, token)``,
    ``list_tasks(device, token, **filter_kws)``, and optionally
    ``prepare_task(device, token, **task_kws)`` and
    ``task_metadata(details, prepared)``, see ``docs/source/cloud_provider.md``.
    """
    f = getattr(_provider_module(provider), name, None)
    if f is None:
        raise NotImplementedError(
            "%s provider does not support %s" % (provider.name, name)
        )
    return f  # type: ignore


def list_providers() -> List[Provider]:
    """
    list all cloud providers that tensorcircuit supports

    :return: list of all supported Provider objects
    :rtype: List[Provider]
    """
    return [get_provider(s) for s in avail_providers]


def set_provider(
    provider: Optional[Union[str, Provider]] = None, set_global: bool = True
) -> Provider:
    """
    set default provider for the program

    :param provider: provider name or Provider instance; defaults to None
        which means the current default provider
    :type provider: Optional[Union[str, Provider]], optional
    :param set_global: whether set, defaults to True,
        if False, equivalent to ``get_provider``
    :type set_global: bool, optional
    :return: the resolved Provider
    :rtype: Provider
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "default_provider", provider)
    return provider


set_provider()
get_provider = partial(set_provider, set_global=False)

default_device = Device.from_name("tencent::simulator:tc")


def set_device(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    set_global: bool = True,
) -> Device:
    """
    set the default device

    :param provider: provider of the device, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: the device, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param set_global: whether set, defaults to True,
        if False, equivalent to ``get_device``, defaults to True
    :type set_global: bool, optional
    :return: the resolved Device
    :rtype: Device
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is None and provider is not None:
        raise ValueError("Please specify the device apart from the provider")
    if device is None:
        device = default_device

    if isinstance(device, str):
        if len(device.split(sep)) > 1:
            provider, device = device.split(sep)
            provider = Provider.from_name(provider)
            device = Device.from_name(device, provider)
        else:
            if provider is None:
                provider = get_provider()
            provider = Provider.from_name(provider)
            device = Device.from_name(device, provider)
    else:
        if provider is None:
            provider = get_provider()
        provider = Provider.from_name(provider)
        device = Device.from_name(device, provider)

    if set_global:
        for module in sys.modules:
            if module.startswith(package_name):
                setattr(sys.modules[module], "default_device", device)
    return device


set_device()
get_device = partial(set_device, set_global=False)

default_store: Optional[TaskStore] = None


def set_store(
    store: Optional[Union[bool, PathLike, TaskStore]] = True,
) -> Optional[TaskStore]:
    """
    Set the default local task store for all following submissions,
    including those inside wrappers such as ``batch_expectation_ps``.
    The ``store`` argument of :py:func:`submit_task` overrides it per call.

    :Example:

    >>> apis.set_store(True)  # record tasks in ~/.tensorcircuit/tasks
    >>> apis.set_store("./my_tasks")  # record tasks in a given directory
    >>> apis.set_store(False)  # stop recording, the default

    :param store: ``True`` for the default directory, a path for a given
        directory, ``False`` or None for no store, defaults to True
    :type store: Optional[Union[bool, PathLike, TaskStore]], optional
    :return: the default task store, None if disabled
    :rtype: Optional[TaskStore]
    """
    global default_store
    default_store = as_store(store)
    return default_store


def get_store() -> Optional[TaskStore]:
    """
    Get the default local task store set by :py:func:`set_store`

    :return: the default task store, None if disabled
    :rtype: Optional[TaskStore]
    """
    return default_store


def b64encode_s(s: str) -> str:
    return b64encode(s.encode("utf-8")).decode("utf-8")


def b64decode_s(s: str) -> str:
    return b64decode(s.encode("utf-8")).decode("utf-8")


def _auth_path() -> str:
    env = os.environ.get("TC_AUTH_PATH")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".tc.auth.json")


def _read_auth_file(path: str) -> Dict[str, str]:
    # atomic-ish read: readers either see the old file or the new one,
    # never a half-written one (writes go through tempfile + os.replace).
    # a missing or corrupt file simply yields an empty dict; we must never
    # raise here, since this runs on module import.
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            file_token = json.load(f)
        return {k: b64decode_s(v) for k, v in file_token.items()}
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("token file loading failure, treat as empty: %s", e)
        return {}


def _write_auth_file(path: str, tokens: Dict[str, str]) -> None:
    # atomic write via tempfile + os.replace, so a crash or a concurrent
    # reader can never observe a truncated/empty file. mkstemp creates the
    # temp file with mode 0600 (ignoring umask), which is the right
    # permission for a credential file on POSIX; on Windows the temp file
    # inherits owner-only ACL from its home directory.
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        logger.warning("token directory creation failure: %s", directory)
        return
    payload = {k: b64encode_s(v) for k, v in tokens.items()}
    fd, tmp = tempfile.mkstemp(prefix=".tc.auth.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("token file writing failure, skip cache saving: %s", e)
        try:
            os.remove(tmp)
        except OSError:
            pass


# Process-local tokens (key -> token), where key is ``"<provider>::"`` or
# ``"<provider>::<device>"``. It holds only tokens set via ``set_token`` in this
# process, never a mirror of the auth file, and is only mutated in place.
saved_token: Dict[str, Any] = {}


def _token_key(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> str:
    if isinstance(provider, str):
        provider = Provider.from_name(provider)
    if device is None:
        if provider is None:
            provider = default_provider
        return provider.name + sep
    device = Device.from_name(device)
    if provider is None:
        provider = device.provider
    if provider is None:
        provider = default_provider
    return provider.name + sep + device.name


def _check_token(token: Any) -> None:
    # never echo the token itself in the error message
    if not isinstance(token, str):
        raise TypeError("token must be a str, got %s" % type(token).__name__)
    if not token.strip():
        raise ValueError("token must be non-empty")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in token):
        raise ValueError("token must not contain control characters such as newlines")


def _preprocess(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Tuple[Provider, Device]:
    """
    Smartly determine the provider and device based on the input

    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device name or Device instance; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: a pair of provider and device after preprocessing
    :rtype: Tuple[Provider, Device]
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is None:
        device = get_device()
    if isinstance(device, str):
        if len(device.split(sep)) > 1:
            device = Device.from_name(device, provider)
        else:
            if provider is None:
                provider = get_provider()
            device = Device.from_name(device, provider)
    if provider is None:
        provider = device.provider
    if isinstance(provider, str):
        provider = Provider.from_name(provider)
    return provider, device  # type: ignore


def set_token(
    token: Optional[str] = None,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    cached: bool = False,
    clear: bool = False,
) -> Dict[str, Any]:
    """
    Set API token for given provider or specifically to given device.
    The token is only kept in the current process unless ``cached=True``,
    which also saves it to the auth file ``~/.tc.auth.json`` (or the path in
    ``TC_AUTH_PATH``). The file is base64 encoded, not encrypted, and is
    readable by the owner only on POSIX.

    :Example:

    >>> apis.set_token("my-token", provider="tencent")  # this process only
    >>> apis.set_token("my-token", provider="tencent", cached=True)  # also on disk
    >>> apis.set_token(provider="tencent", clear=True)  # remove tencent tokens

    :param token: the API token, defaults to None, which returns all tokens
        saved in memory and on the disk
    :type token: Optional[str], optional
    :param provider: provider scope of the token; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device scope of the token (more specific than provider);
        defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param cached: whether also save the token on the disk, defaults to False
    :type cached: bool, optional
    :param clear: if True, remove saved tokens both in memory and on the disk:
        the given device token, all tokens of the given provider,
        or all tokens if no scope is given, defaults to False
    :type clear: bool, optional
    :return: the tokens saved in memory of this process
    :rtype: Dict[str, Any]
    """
    authpath = _auth_path()
    if clear is True:
        if provider is None and device is None:
            saved_token.clear()
            if os.path.exists(authpath):
                try:
                    os.remove(authpath)
                except OSError as e:
                    logger.warning("token file removal failure: %s", e)
            return saved_token
        key = _token_key(provider, device)

        def in_scope(k: str) -> bool:
            return k == key or (device is None and k.startswith(key))

        for k in [k for k in saved_token if in_scope(k)]:
            del saved_token[k]
        file_token = _read_auth_file(authpath)
        kept = {k: v for k, v in file_token.items() if not in_scope(k)}
        if len(kept) < len(file_token):
            _write_auth_file(authpath, kept)
        return saved_token
    if token is None:
        # pure read, never write back: rewriting a stale snapshot on import
        # used to truncate the auth file under multiprocessing
        file_token = _read_auth_file(authpath)
        file_token.update(saved_token)
        return file_token
    _check_token(token)
    key = _token_key(provider, device)
    saved_token[key] = token
    if cached:
        file_token = _read_auth_file(authpath)
        file_token[key] = token
        _write_auth_file(authpath, file_token)
    return saved_token


set_token()


def get_token(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Optional[str]:
    """
    Get API token set for given provider or device.
    The lookup order is: tokens set by ``set_token`` in this process,
    the ``TC_TOKEN_<PROVIDER_NAME>`` environment variable (uppercased,
    e.g. ``TC_TOKEN_TENCENT``), and the auth file. Within the process and
    the auth file, a device token takes precedence over the provider token.

    :param provider: provider scope to look up; defaults to the current
        default provider
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device scope to look up; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: the matching token, or None if no token is set
    :rtype: Optional[str]
    """
    if provider is None:
        provider = get_provider()
    provider = Provider.from_name(provider)
    keys = [provider.name + sep]
    if device is not None:
        device = Device.from_name(device, provider)
        keys.insert(0, provider.name + sep + device.name)
    for key in keys:
        if key in saved_token:
            return saved_token[key]  # type: ignore
    env_token = os.environ.get("TC_TOKEN_" + provider.name.upper())
    if env_token:
        return env_token
    file_token = _read_auth_file(_auth_path())
    for key in keys:
        if key in file_token:
            return file_token[key]
    return None


# token json structure
# {"tencent::": token1, "tencent::20xmon":  token2}


def list_devices(
    provider: Optional[Union[str, Provider]] = None,
    token: Optional[str] = None,
    **kws: Any,
) -> List[Device]:
    """
    List all devices under a provider

    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :return: list of Device objects under the provider
    :rtype: Any
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()
    return _provider_api(provider, "list_devices")(token, **kws)  # type: ignore


def list_properties(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List properties of a given device

    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device name or Device instance; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :return: Propeties dict
    :rtype: Dict[str, Any]
    """
    provider, device = _preprocess(provider, device)

    if token is None:
        token = device.get_token()  # type: ignore
    return _provider_api(provider, "list_properties")(device, token)  # type: ignore


def get_task(
    taskid: str,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Task:
    """
    Get ``Task`` object from task string, the binding device can also be provided

    :param taskid: task id string, optionally prefixed with the device via
        the sep2 separator
    :type taskid: str
    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device name or Device instance; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: the resolved Task object
    :rtype: Task
    """
    if provider is not None and device is None:
        provider, device = None, provider
    if device is not None:  # device can be None for identify tasks
        device = Device.from_name(device, provider)
    elif len(taskid.split(sep2)) > 1:
        device = Device(taskid.split(sep2)[0])
        taskid = taskid.split(sep2)[1]
    return Task(taskid, device=device)


def get_task_details(
    taskid: Union[str, Task], token: Optional[str] = None, prettify: bool = False
) -> Dict[str, Any]:
    """
    Get task details dict given task id

    :param taskid: task id string or Task object
    :type taskid: Union[str, Task]
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :param prettify: whether make the returned dict more readable and more phythonic,
        defaults to False
    :type prettify: bool
    :return: task details dict (raw or prettified)
    :rtype: Dict[str, Any]
    """
    if isinstance(taskid, str):
        task = Task(taskid)
    else:
        task = taskid
    if task.device is not None:
        device = task.device
    else:
        device = default_device
    if token is None:
        token = device.get_token()
    details = _provider_api(device.provider, "get_task_details")
    return details(task, device, token, prettify)  # type: ignore


def get_task_metadata(task: Task, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the provider-independent metadata of the task details, i.e.
    ``task.details()["tensorcircuit"]`` without the archive status, see
    :py:meth:`Task.details`. The provider module gives the values it knows from
    its own details and :py:attr:`Task.prepared`, empty if unknown, by
    ``task_metadata(details, prepared)``, and the other values are None.

    :param task: the task
    :type task: Task
    :param details: the task details by :py:func:`get_task_details`
    :type details: Dict[str, Any]
    :return: the metadata
    :rtype: Dict[str, Any]
    """
    meta: Dict[str, Any] = dict.fromkeys(metadata_keys)
    if task.device is not None:
        f = getattr(_provider_module(task.device.provider), "task_metadata", None)
        if f is not None:
            meta.update(f(details, task.prepared or {}))
    return dict(schema_version=1, **meta)


def submit_task(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    store: Optional[Union[bool, PathLike, TaskStore]] = None,
    **task_kws: Any,
) -> List[Task]:
    """
    submit task to the cloud platform, batch submission default enabled

    .. seealso::

        :py:func:`tensorcircuit.cloud.tencent.submit_task`

    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device name or Device instance; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :param store: whether record the tasks in a local task store: ``True`` for
        the default directory, a path for a given directory, or ``False``;
        defaults to None, which follows :py:func:`set_store`
    :type store: Optional[Union[bool, PathLike, TaskStore]], optional
    :param task_kws: all necessary keywords arguments for task submission,
        see detailed API in each provider backend:
        1. tencent - :py:func:`tensorcircuit.cloud.tencent.submit_task`
        2. tianyan - :py:func:`tensorcircuit.cloud.tianyan.submit_task`
        3. lqcloud - :py:func:`tensorcircuit.cloud.lqcloud.submit_task`
    :type task_kws: Any
    :return: Task object or list of Task for batch submission
    :rtype: List[Task]
    """
    provider, device = _preprocess(provider, device)
    # resolve the provider before recording anything in the store
    submit = _provider_api(provider, "submit_task")
    if token is None:
        token = device.get_token()  # type: ignore
    ts = default_store if store is None else as_store(store)

    # providers take an explicit source over the circuit
    source = task_kws.get("source")
    items = task_kws.get("circuit") if source is None else source
    batch = is_sequence(items)
    items = list(items) if batch else [items]  # type: ignore
    prepare = getattr(_provider_module(provider), "prepare_task", None)
    if prepare is None:
        send_kws, prepared = task_kws, [_prepared(c) for c in items]
    else:
        # compile locally first, so the intents hold what is prepared for each circuit
        send_kws, prepared = prepare(device, token, **task_kws)
    records = []
    if ts is not None:
        records = make_records(
            provider.name,
            device.name,
            _task_requests(items, task_kws),
            batch,
            _provider_modules[provider.name][1],
        )
    return _send_tasks(  # type: ignore
        ts, records, prepared, lambda: submit(device, token, **send_kws)
    )


def _prepared(c: Any) -> Dict[str, Any]:
    # what is prepared for a circuit by providers without ``prepare_task``
    mapping = (
        c.get_positional_logical_mapping() if isinstance(c, AbstractCircuit) else None
    )
    return {"positional_logical_mapping": mapping}


def _send_tasks(
    ts: Optional[TaskStore],
    records: List[Dict[str, Any]],
    prepared: Sequence[Optional[Dict[str, Any]]],
    send: Callable[[], Any],
) -> Any:
    """
    Send paid tasks once, recording them in the store if any: the submission
    intents are saved before sending, kept as ``unknown`` if the tasks may have
    been accepted, and deleted for the tasks certainly not submitted.
    Once the request is sent, ``send`` returns the accepted tasks, where each
    missing circuit is explicitly rejected, or raises ``SubmissionRejected``
    or ``SubmissionUnknown``; any other failure happens before sending.
    ``prepared`` holds what is prepared for each circuit, see :py:attr:`Task.prepared`.
    """
    for rec, p in zip(records, prepared):
        rec["prepared"] = p
    if ts is not None:
        try_store(ts.save, records)
    try:
        r = send()
    except unknown_submission_errors as e:
        if ts is not None:
            for rec in records:
                rec["submission"] = {"status": "unknown", "error": repr(e)}
            try_store(ts.save, records)
        raise
    except Exception:
        if ts is not None:
            try_store(ts.delete, [rec["local_id"] for rec in records])
        raise
    tasks = r if isinstance(r, list) else [r]
    for i, t in enumerate(tasks):
        if t.circuit_index is None:
            t.circuit_index = i
        if t.prepared is None:
            t.prepared = prepared[t.circuit_index]
    if ts is not None:
        bound = {t.circuit_index: t for t in tasks}
        accepted = [rec for rec in records if rec["circuit_index"] in bound]
        for rec in accepted:
            t = bound[rec["circuit_index"]]
            rec["remote_id"] = t.id_
            rec["prepared"] = t.prepared
            rec["submission"] = {"status": "submitted", "error": None}
        # circuits explicitly rejected by the provider in a batch
        try_store(
            ts.delete,
            [rec["local_id"] for rec in records if rec["circuit_index"] not in bound],
        )
        error = try_store(ts.save, accepted)
        for rec in accepted:
            bound[rec["circuit_index"]]._bind_store(ts, rec["local_id"], error)
    return r


def _circuit_record(c: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # a circuit that cannot be archived must never block a valid submission
    if not isinstance(c, AbstractCircuit):
        return None, None
    try:
        return circuit_record(c), None
    except Exception as e:
        logger.warning("task store cannot serialize the circuit: %r", e)
        return None, repr(e)


def _task_requests(items: List[Any], task_kws: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots = task_kws.get("shots")
    options = {
        k: v for k, v in task_kws.items() if k not in ["circuit", "source", "shots"]
    }
    requests = []
    for i, c in enumerate(items):
        circuit, error = _circuit_record(c)
        requests.append(
            {
                "circuit": circuit,
                "circuit_error": error,
                "circuit_type": None if c is None else type(c).__name__,
                "source": c if isinstance(c, str) else None,
                "shots": shots[i] if is_sequence(shots) else shots,  # type: ignore
                "options": options,
            }
        )
    return requests


def resubmit_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
    store: Optional[Union[bool, PathLike, TaskStore]] = None,
) -> Task:
    """
    Rerun the given task on the provider as a new paid task, which is sent once
    and recorded with the original record as its parent if a store is active

    :param task: task id string or Task object to rerun
    :type task: Optional[Union[str, Task]]
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :param store: the task store, see :py:func:`submit_task`; defaults to None,
        i.e. the store of ``task`` if any, otherwise the global store
    :type store: Optional[Union[bool, PathLike, TaskStore]], optional
    :return: the newly submitted Task
    :rtype: Task
    """
    if isinstance(task, str):
        task = get_task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    provider = device.provider
    resubmit = _provider_api(provider, "resubmit_task")
    # the rerun is recorded where the original task is, unless specified
    ts = as_store(store) if store is not None else (task._store or default_store)  # type: ignore
    parent = None if ts is None else _parent_record(ts, task, provider.name)  # type: ignore
    prepared = task.prepared  # type: ignore
    if prepared is None and parent:
        # the rerun sends the same, e.g. the compiled program, which the record holds
        prepared = parent.get("prepared")
    records = []
    if ts is not None:
        request = dict(parent["request"] or {}) if parent else {}
        request["resubmit_of"] = task.id_  # type: ignore
        records = make_records(
            provider.name,
            device.name,
            [request],
            False,
            _provider_modules[provider.name][1],
        )
        if parent:
            records[0]["parent_id"] = parent["local_id"]
    if prepared is not None and task.prepared is None:  # type: ignore
        # the provider reruns from what the task holds, without changing the given one
        task = copy(task)
        task.prepared = prepared  # type: ignore
    return _send_tasks(  # type: ignore
        ts, records, [prepared], lambda: resubmit(task, token)
    )


def _parent_record(
    ts: TaskStore, task: Task, provider: str
) -> Optional[Dict[str, Any]]:
    # the record of the task to rerun, a store failure never blocks the rerun
    try:
        record = ts.get(task.local_id) if task.local_id is not None else None
        if record is None:
            record = ts.get(remote_id=task.id_, provider=provider)
    except store_errors as e:
        logger.warning("task store read failure: %s", e)
        return None
    if record is None or record["integrity"] not in ["complete", "pending"]:
        return None
    return record


def remove_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
) -> Task:
    if isinstance(task, str):
        task = Task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    return _provider_api(device.provider, "remove_task")(task, token)  # type: ignore


def list_tasks(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **filter_kws: Any,
) -> List[Task]:
    """
    List tasks based on given filters

    :param provider: provider name or Provider instance; defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: device name or Device instance; defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: API token for the provider; defaults to None which uses
        the saved token
    :type token: Optional[str], optional
    :return: list of task object that satisfy these filter criteria
    :rtype: List[Task]
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()  # type: ignore
    if device is not None:
        device = Device.from_name(device)
    return _provider_api(provider, "list_tasks")(device, token, **filter_kws)  # type: ignore
