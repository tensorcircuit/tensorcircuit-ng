"""
main entrypoints of cloud module
"""

from typing import Any, List, Optional, Dict, Union, Tuple
from base64 import b64decode, b64encode
from functools import partial
import json
import os
import sys
import tempfile
import logging

from .abstraction import Provider, Device, Task, sep, sep2

logger = logging.getLogger(__name__)


try:
    from . import tencent  # type: ignore
except (ImportError, ModuleNotFoundError):
    logger.info("fail to load cloud provider module: tencent")

try:
    from . import local
except (ImportError, ModuleNotFoundError):
    logger.info("fail to load cloud provider module: local")

try:
    from . import quafu_provider
except (ImportError, ModuleNotFoundError):
    logger.info("fail to load cloud provider module: quafu")

package_name = "tensorcircuit"
thismodule = sys.modules[__name__]


default_provider = Provider.from_name("tencent")
avail_providers = ["tencent", "local", "quafu"]


def list_providers() -> List[Provider]:
    """
    list all cloud providers that tensorcircuit supports

    :return: _description_
    :rtype: List[Provider]
    """
    return [get_provider(s) for s in avail_providers]


def set_provider(
    provider: Optional[Union[str, Provider]] = None, set_global: bool = True
) -> Provider:
    """
    set default provider for the program

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param set_global: whether set, defaults to True,
        if False, equivalent to ``get_provider``
    :type set_global: bool, optional
    :return: _description_
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
    :return: _description_
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


# Process-local session token overrides (key -> token), where key is
# ``"<provider>::"`` or ``"<provider>::<device>"``. This is the session layer
# on top of the disk-backed auth file: it holds only tokens explicitly set via
# ``set_token`` in this process (including ``cached=False`` writes), never a
# mirror of disk. Readers must not rebind this global; ``get_token`` falls back
# to disk when a key is absent here, so in-place mutations (e.g. ``patch.dict``
# in tests) survive across ``get_token`` calls.
saved_token: Dict[str, Any] = {}


def _preprocess(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Tuple[Provider, Device]:
    """
    Smartly determine the provider and device based on the input

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
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
    cached: bool = True,
    clear: bool = False,
) -> Dict[str, Any]:
    """
    Set API token for given provider or specifically to given device

    :param token: the API token, defaults to None
    :type token: Optional[str], optional
    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param cached: whether save on the disk, defaults to True
    :type cached: bool, optional
    :param clear: if True, clear all token saved, defaults to False
    :type clear: bool, optional
    :return: _description_
    :rtype: Dict[str, Any]
    """
    global saved_token
    authpath = _auth_path()
    if clear is True:
        saved_token = {}
        if cached and os.path.exists(authpath):
            try:
                os.remove(authpath)
            except OSError as e:
                logger.warning("token file removal failure: %s", e)
        return saved_token
    if token is None:
        # pure read: return the merged view of disk + session overrides.
        # Never write back to disk here: writing back on import / get_token is
        # what used to truncate the auth file under multiprocessing (each
        # worker re-dumped a stale snapshot).
        file_token = _read_auth_file(authpath)
        file_token.update(saved_token)
        return file_token
    # with token: write the session override layer first (so get_token can read
    # it even when cached=False), then optionally persist to disk.
    if isinstance(provider, str):
        provider = Provider.from_name(provider)
    if device is None:
        if provider is None:
            provider = default_provider
        key = provider.name + sep
    else:
        device = Device.from_name(device)
        if provider is None:
            provider = device.provider  # type: ignore
        if provider is None:
            provider = default_provider
        key = provider.name + sep + device.name  # type: ignore
    saved_token[key] = token
    if cached:
        # merge the new entry into the freshest on-disk state and persist
        # atomically, so concurrent processes can't clobber each other.
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
    Get API token setted for given provider or device,
    if no device token saved, the corresponding provider tken is returned.
    Environment variables take precedence and are never persisted: a single
    token can be supplied via ``TC_TOKEN``, or a per-provider token via
    ``TC_TOKEN_<PROVIDER_NAME>`` (uppercased), e.g. ``TC_TOKEN_TENCENT``.

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: _description_
    :rtype: Optional[str]
    """
    if provider is None:
        provider = get_provider()
    provider = Provider.from_name(provider)
    target = provider.name + sep
    if device is not None:
        device = Device.from_name(device, provider)
        target = target + device.name
    # 1. env var override (process-scoped, never persisted): single token,
    #    or per-provider token.
    env_token = os.environ.get("TC_TOKEN")
    if env_token is None:
        env_token = os.environ.get("TC_TOKEN_" + provider.name.upper())
    if env_token is not None:
        return env_token
    # 2. session override (in-memory, process-scoped, set via set_token).
    #    Read-only here: never rebind the global, so in-place mutations
    #    (e.g. patch.dict in tests) survive across get_token calls, and
    #    cached=False writes remain visible to readers.
    if target in saved_token:
        return saved_token[target]  # type: ignore
    # 3. disk (persistent, shared across processes); read-only.
    file_token = _read_auth_file(_auth_path())
    if target in file_token:
        return file_token[target]  # type: ignore
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

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: _description_
    :rtype: Any
    """
    if provider is None:
        provider = default_provider
    provider = Provider.from_name(provider)
    if token is None:
        token = provider.get_token()
    if provider.name == "tencent":
        return tencent.list_devices(token, **kws)  # type: ignore
    elif provider.name == "local":
        return local.list_devices(token, **kws)
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)


def list_properties(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List properties of a given device

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: Propeties dict
    :rtype: Dict[str, Any]
    """
    provider, device = _preprocess(provider, device)

    if token is None:
        token = device.get_token()  # type: ignore
    if provider.name == "tencent":  # type: ignore
        return tencent.list_properties(device, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def get_task(
    taskid: str,
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
) -> Task:
    """
    Get ``Task`` object from task string, the binding device can also be provided

    :param taskid: _description_
    :type taskid: str
    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :return: _description_
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

    :param taskid: _description_
    :type taskid: Union[str, Task]
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :param prettify: whether make the returned dict more readable and more phythonic,
        defaults to False
    :type prettify: bool
    :return: _description_
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
    provider = device.provider

    if provider.name == "tencent":
        return tencent.get_task_details(task, device, token, prettify)  # type: ignore
    elif provider.name == "local":
        return local.get_task_details(task, device, token, prettify)  # type: ignore
    elif provider.name == "quafu":
        return quafu_provider.get_task_details(task, device, token, prettify)  # type: ignore

    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def submit_task(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **task_kws: Any,
) -> List[Task]:
    """
    submit task to the cloud platform, batch submission default enabled

    .. seealso::

        :py:func:`tensorcircuit.cloud.tencent.submit_task`

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :param task_kws: all necessary keywords arguments for task submission,
        see detailed API in each provider backend:
        1. tencent - :py:func:`tensorcircuit.cloud.tencent.submit_task`
    :type task_kws: Any
    :return: Task object or list of Task for batch submission
    :rtype: List[Task]
    """
    provider, device = _preprocess(provider, device)

    if token is None:
        token = device.get_token()  # type: ignore
    if provider.name == "tencent":  # type: ignore
        return tencent.submit_task(device, token, **task_kws)  # type: ignore
    elif provider.name == "local":  # type: ignore
        return local.submit_task(device, token, **task_kws)  # type: ignore
    elif provider.name == "quafu":  # type: ignore
        return quafu_provider.submit_task(device, token, **task_kws)  # type: ignore
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def resubmit_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
) -> Task:
    """
    Rerun the given task

    :param task: _description_
    :type task: Optional[Union[str, Task]]
    :param token: _description_, defaults to None
    :type token: Optional[str], optional
    :return: _description_
    :rtype: Task
    """
    if isinstance(task, str):
        task = Task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    provider = device.provider

    if provider.name == "tencent":  # type: ignore
        return tencent.resubmit_task(task, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def remove_task(
    task: Optional[Union[str, Task]],
    token: Optional[str] = None,
) -> Task:
    if isinstance(task, str):
        task = Task(task)
    device = task.get_device()  # type: ignore
    if token is None:
        token = device.get_token()
    provider = device.provider

    if provider.name == "tencent":  # type: ignore
        return tencent.remove_task(task, token)  # type: ignore
    elif provider.name == "local":
        raise ValueError("Unsupported method for local backend")
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore


def list_tasks(
    provider: Optional[Union[str, Provider]] = None,
    device: Optional[Union[str, Device]] = None,
    token: Optional[str] = None,
    **filter_kws: Any,
) -> List[Task]:
    """
    List tasks based on given filters

    :param provider: _description_, defaults to None
    :type provider: Optional[Union[str, Provider]], optional
    :param device: _description_, defaults to None
    :type device: Optional[Union[str, Device]], optional
    :param token: _description_, defaults to None
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
    if provider.name == "tencent":  # type: ignore
        return tencent.list_tasks(device, token, **filter_kws)  # type: ignore
    elif provider.name == "local":  # type: ignore
        return local.list_tasks(device, token, **filter_kws)  # type: ignore
    else:
        raise ValueError("Unsupported provider: %s" % provider.name)  # type: ignore
