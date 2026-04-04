"""ZX graph simplification wrapper."""

from typing import Any
import pyzx_param as pyzx


def full_reduce(g: Any, param_safe: bool = True) -> None:
    """
    Apply PyZX full_reduce to the graph.

    :param g: PyZX Graph
    :param param_safe: Whether to use parameter-safe simplification
    """
    pyzx.full_reduce(g, paramSafe=param_safe)


def teleport_reduce(g: Any) -> Any:
    """
    Apply PyZX teleport_reduce to the graph.
    Note: returns a new graph.
    """
    return pyzx.teleport_reduce(g)


def t_count(g: Any) -> int:
    """
    Return the T-count of the graph.
    """
    return pyzx.tcount(g)  # type: ignore[no-any-return]
