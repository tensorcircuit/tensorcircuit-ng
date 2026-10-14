"""
Some common graphs and lattices
"""

# pylint: disable=invalid-name

from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

Graph = Any


def Line1D(
    n: int,
    node_weight: Optional[Union[float, Sequence[float], np.ndarray[Any, Any]]] = None,
    edge_weight: Optional[Union[float, Sequence[float], np.ndarray[Any, Any]]] = None,
    pbc: bool = True,
) -> Graph:
    """
    1D chain with ``n`` sites

    :param n: number of sites in the chain; periodic chains require at least 2
    :type n: int
    :param node_weight: scalar weight broadcast to all sites, or a sequence or
        one-dimensional array with at least ``n`` weights in site order.
        Extra entries are ignored; defaults to zero.
    :type node_weight: Optional[Union[float, Sequence[float], numpy.ndarray]]
    :param edge_weight: scalar weight broadcast to all bonds, or a sequence or
        one-dimensional array with at least ``n - 1`` weights in chain order.
        For periodic boundaries, entry ``n - 1`` weights the closing bond
        ``(n - 1, 0)``; if only ``n - 1`` entries are supplied, the last weight
        is reused. Extra entries are ignored; defaults to one. For a two-site
        periodic chain, the two bond weights are summed on its single edge.
    :type edge_weight: Optional[Union[float, Sequence[float], numpy.ndarray]]
    :param pbc: whether to use periodic boundary conditions (close the chain into a ring), defaults to True
    :type pbc: bool, optional
    :return: the 1D chain as a networkx graph
    :rtype: Graph
    :raises ValueError: if ``pbc`` is true and ``n < 2``
    """

    if pbc and n < 2:
        raise ValueError("Periodic Line1D requires at least two sites")

    g = nx.Graph()
    if edge_weight is None:
        edge_weight = 1.0
    if not isinstance(edge_weight, Sequence) and np.ndim(edge_weight) == 0:
        edge_weight = [edge_weight] * n  # type: ignore[list-item]
    if node_weight is None:
        node_weight = 0.0
    if not isinstance(node_weight, Sequence) and np.ndim(node_weight) == 0:
        node_weight = [node_weight] * n  # type: ignore[list-item]
    for i in range(n):
        g.add_node(i, weight=node_weight[i])  # type: ignore[index]
    for i in range(n - 1):
        g.add_edge(i, i + 1, weight=edge_weight[i])  # type: ignore[index]
    if pbc:
        weight = edge_weight[min(n, len(edge_weight)) - 1]  # type: ignore[index, arg-type]
        if g.has_edge(n - 1, 0):
            g[n - 1][0]["weight"] = g[n - 1][0]["weight"] + weight
        else:
            g.add_edge(n - 1, 0, weight=weight)
    return g


def Even1D(n: int, s: int = 0) -> Graph:
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, weight=1.0)
    for i in range(s, n, 2):
        g.add_edge(i, (i + 1) % n, weight=1.0)
    return g


Odd1D = partial(Even1D, s=1)


class Grid2DCoord:
    """
    Two-dimensional grid lattice
    """

    def __init__(self, n: int, m: int):
        """

        :param n: number of rows
        :type n: int
        :param m: number of cols
        :type m: int
        """
        # row first
        self.m = m
        self.n = n
        self.mn = m * n

    def one2two(self, i: int) -> Tuple[int, int]:
        x = i // self.n
        y = i % self.n
        return x, y

    def two2one(self, x: int, y: int) -> int:
        return x * self.n + y

    def all_rows(self, pbc: bool = False) -> Sequence[Tuple[int, int]]:
        """
        return all row edge with 1d index encoding

        :param pbc: whether to include pbc edges (periodic boundary condition),
            defaults to False
        :type pbc: bool, optional
        :return: list of row edge
        :rtype: Sequence[Tuple[int, int]]
        """
        r = []
        for i in range(self.mn):
            if (i + 1) % self.n != 0:
                r.append((i, i + 1))
            elif pbc:
                r.append((i, i - self.n + 1))
        return r

    def all_cols(self, pbc: bool = False) -> Sequence[Tuple[int, int]]:
        """
        return all col edge with 1d index encoding

        :param pbc: whether to include pbc edges (periodic boundary condition),
            defaults to False
        :type pbc: bool, optional
        :return: list of col edge
        :rtype: Sequence[Tuple[int, int]]
        """
        r = []
        for i in range(self.mn):
            if i + self.n < self.mn:
                r.append((i, i + self.n))
            elif pbc:
                r.append((i, i - (self.m - 1) * self.n))
        return r

    def lattice_graph(self, pbc: bool = True) -> Graph:
        """
        Get the 2D grid lattice in ``nx.Graph`` format

        :param pbc: whether to include pbc edges (periodic boundary condition),
            defaults to True
        :type pbc: bool, optional
        :return: the 2D grid lattice as a networkx graph
        :rtype: Graph
        """
        g = nx.Graph()
        for i in range(self.mn):
            g.add_node(i, weight=0)
        for i in range(self.mn):
            x, y = self.one2two(i)
            if pbc is False and x - 1 < 0:
                pass
            else:
                g.add_edge(i, self.two2one((x - 1) % self.m, y), weight=1)
            if pbc is False and y - 1 < 0:
                pass
            else:
                g.add_edge(i, self.two2one(x, (y - 1) % self.n), weight=1)
        return g
