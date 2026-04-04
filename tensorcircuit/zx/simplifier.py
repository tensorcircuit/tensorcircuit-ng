"""ZX graph simplification wrapper."""

from typing import Any
import pyzx_param as pyzx


def full_reduce(g: Any, param_safe: bool = True) -> None:
    """
    Apply PyZX full_reduce to the graph.

    :param g: The ZX graph.
    :type g: Any
    :param param_safe: Whether to use parameter-safe simplification, defaults to True.
    :type param_safe: bool, optional
    """
    pyzx.full_reduce(g, paramSafe=param_safe)


def teleport_reduce(g: Any) -> Any:
    """
    Apply PyZX teleport_reduce to the graph.

    :param g: The ZX graph.
    :type g: Any
    :return: A new simplified graph.
    :rtype: Any
    """
    return pyzx.teleport_reduce(g)


def t_count(g: Any) -> int:
    """
    Return the T-count of the graph.

    :param g: The ZX graph.
    :type g: Any
    :return: Number of T or T-like vertices.
    :rtype: int
    """
    return pyzx.tcount(g)  # type: ignore[no-any-return]
