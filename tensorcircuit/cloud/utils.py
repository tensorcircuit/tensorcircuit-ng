"""
utility functions for cloud connection
"""

from typing import Any, Callable, Optional
from functools import wraps
import inspect
import logging
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests

# from simplejson.errors import JSONDecodeError

logger = logging.getLogger(__name__)
thismodule = sys.modules[__name__]


class HttpStatusError(Exception):
    """
    Used when the return request has http code beyond 200
    """

    pass


class SubmissionUnknown(Exception):
    """
    Raised when a request creating paid tasks fails after it may have reached
    the server, so the tasks may have been accepted; it is never retried
    """

    pass


class SubmissionRejected(ValueError):
    """
    Raised when the provider explicitly rejects a request creating paid tasks,
    so the tasks are certainly not submitted
    """

    pass


# TODO(@refraction-ray): whether an exception hierarchy for tc is necessary?
connection_errors = (
    ConnectionResetError,
    HttpStatusError,
    requests.exceptions.RequestException,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    ValueError,
    # JSONDecodeError,
)


def set_proxy(proxy: Optional[str] = None) -> None:
    """
    :param proxy: str. format as "http://user:passwd@host:port" user passwd part can be omitted if not set.
        None for turning off the proxy.
    :return:
    """
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        setattr(thismodule, "proxy", proxy)
    else:
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        setattr(thismodule, "proxy", None)


def redact_url(url: str) -> str:
    """
    Hide the ``user:password@`` part of a URL, e.g., for logging proxies.

    :param url: URL possibly carrying credentials
    :type url: str
    :return: URL with the credentials replaced by ``***``
    :rtype: str
    """
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc="***@" + parts.netloc.rsplit("@", 1)[1]))


def reconnect(tries: int = 5, timeout: int = 12) -> Callable[..., Any]:
    # wrapper originally designed in xalpha by @refraction-ray
    # https://github.com/refraction-ray/xalpha
    def robustify(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def wrapper(*args: Any, **kws: Any) -> Any:
            if getattr(thismodule, "proxy", None):
                kws["proxies"] = {
                    "http": getattr(thismodule, "proxy"),
                    "https": getattr(thismodule, "proxy"),
                }
                logger.debug("Using proxy %s", redact_url(getattr(thismodule, "proxy")))
            kws["timeout"] = timeout
            if args:
                url = args[0]
            else:
                url = kws.get("url", "")
            headers = kws.get("headers", {})
            if (not headers.get("user-agent", None)) and (
                not headers.get("User-Agent", None)
            ):
                headers["user-agent"] = "Mozilla/5.0"
            kws["headers"] = headers
            for count in range(tries):
                try:
                    logger.debug(
                        "Fetching url: %s . Inside function `%s`"
                        % (url, inspect.stack()[1].function)
                    )
                    r = f(*args, **kws)
                    if (
                        getattr(r, "status_code", 200) != 200
                    ):  # in case r is a json dict
                        raise HttpStatusError
                    return r
                except connection_errors as e:
                    if count == tries - 1:
                        logger.error(
                            "Still wrong at fetching url: %s. after %s tries."
                            % (url, tries)
                        )
                        logger.error("Fails due to %r", e)
                        raise e
                    logger.warning("Fails at fetching url: %s. Try again." % url)
                    time.sleep(0.5 * count)

        return wrapper

    return robustify


rget = reconnect()(requests.get)
rpost = reconnect()(requests.post)


@reconnect()
def rget_json(*args: Any, **kws: Any) -> Any:
    r = requests.get(*args, **kws)
    return r.json()


@reconnect()
def rpost_json(*args: Any, **kws: Any) -> Any:
    r = requests.post(*args, **kws)
    return r.json()


@reconnect(tries=1, timeout=60)
def _rpost_json_once(*args: Any, **kws: Any) -> Any:
    r = requests.post(*args, **kws)
    return r.json()


def rpost_json_once(*args: Any, **kws: Any) -> Any:
    """
    Post a request creating paid tasks only once with a longer timeout.
    Any failure of the request itself raises :py:class:`SubmissionUnknown`,
    while an error in a parsed response is left to the caller.
    """
    try:
        return _rpost_json_once(*args, **kws)
    except connection_errors as e:
        raise SubmissionUnknown(repr(e)) from e


# failures after which submitted tasks may have been accepted
unknown_submission_errors = (
    SubmissionUnknown,
    requests.exceptions.RequestException,
    ConnectionError,
    TimeoutError,
)
