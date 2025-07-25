"""
dict related functionalities
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np


Tensor = Any
ct = Dict[str, int]


def reverse_count(count: ct) -> ct:
    """
    Reverse the bit string keys in a count dictionary.

    :param count: A dictionary mapping bit strings to counts
    :type count: ct
    :return: A new dictionary with reversed bit string keys
    :rtype: ct

    :Example:

    >>> reverse_count({"01": 10, "10": 20})
    {'10': 10, '01': 20}
    """
    ncount = {}
    for k, v in count.items():
        ncount[k[::-1]] = v
    return ncount


def sort_count(count: ct) -> ct:
    """
    Sort the count dictionary by counts in descending order.

    :param count: A dictionary mapping bit strings to counts
    :type count: ct
    :return: A new dictionary sorted by count values (descending)
    :rtype: ct

    :Example:

    >>> sort_count({"00": 5, "01": 15, "10": 10})
    {'01': 15, '10': 10, '00': 5}
    """
    return {k: v for k, v in sorted(count.items(), key=lambda item: -item[1])}


def normalized_count(count: ct) -> Dict[str, float]:
    """
    Normalize the count dictionary to represent probabilities.

    :param count: A dictionary mapping bit strings to counts
    :type count: ct
    :return: A new dictionary with probabilities instead of counts
    :rtype: Dict[str, float]

    :Example:

    >>> normalized_count({"00": 5, "01": 15})
    {'00': 0.25, '01': 0.75}
    """
    shots = sum([v for k, v in count.items()])
    return {k: v / shots for k, v in count.items()}


def marginal_count(count: ct, keep_list: Sequence[int]) -> ct:
    """
    Compute the marginal distribution of a count dictionary over specified qubits.

    :param count: A dictionary mapping bit strings to counts
    :type count: ct
    :param keep_list: List of qubit indices to keep in the marginal distribution
    :type keep_list: Sequence[int]
    :return: A new count dictionary with marginal distribution
    :rtype: ct

    :Example:

    >>> marginal_count({"001": 10, "110": 20}, [0, 2])
    {'01': 10, '10': 20}
    """
    import qiskit

    count = reverse_count(count)
    ncount = qiskit.result.utils.marginal_distribution(count, keep_list)
    return reverse_count(ncount)


def count2vec(count: ct, normalization: bool = True) -> Tensor:
    """
    Convert count dictionary to probability vector.

    :param count: A dictionary mapping bit strings to counts
    :type count: ct
    :param normalization: Whether to normalize the counts to probabilities, defaults to True
    :type normalization: bool, optional
    :return: Probability vector as numpy array
    :rtype: Tensor

    :Example:

    >>> count2vec({"00": 2, "10": 3, "11": 5})
    array([0.2, 0. , 0.3, 0.5])
    """
    nqubit = len(list(count.keys())[0])
    probability = [0] * 2**nqubit
    shots = sum([v for k, v in count.items()])
    for k, v in count.items():
        if normalization is True:
            v /= shots  # type: ignore
        probability[int(k, 2)] = v
    return np.array(probability)


def vec2count(vec: Tensor, prune: bool = False) -> ct:
    """
    Convert probability vector to count dictionary.

    :param vec: Probability vector
    :type vec: Tensor
    :param prune: Whether to remove near-zero probabilities, defaults to False
    :type prune: bool, optional
    :return: Count dictionary
    :rtype: ct

    :Example:

    >>> vec2count(np.array([0.2, 0.3, 0.1, 0.4]))
    {'00': 0.2, '01': 0.3, '10': 0.1, '11': 0.4}
    """
    from ..quantum import count_vector2dict

    if isinstance(vec, list):
        vec = np.array(vec)
    n = int(np.log(vec.shape[0]) / np.log(2) + 1e-9)
    c = count_vector2dict(vec, n, key="bin")
    if prune is True:
        nc = c.copy()
        for k, v in c.items():
            if np.abs(v) < 1e-8:
                del nc[k]
        return nc
    return c


def kl_divergence(c1: ct, c2: ct) -> float:
    """
    Compute the Kullback-Leibler divergence between two count distributions.

    :param c1: First count dictionary
    :type c1: ct
    :param c2: Second count dictionary
    :type c2: ct
    :return: KL divergence value
    :rtype: float
    """
    eps = 1e-4  # typical value for inverse of the total shots
    c1 = normalized_count(c1)  # type: ignore
    c2 = normalized_count(c2)  # type: ignore
    kl = 0
    for k, v in c1.items():
        kl += v * (np.log(v) - np.log(c2.get(k, eps)))
    return kl


def expectation(
    count: ct, z: Optional[Sequence[int]] = None, diagonal_op: Optional[Tensor] = None
) -> float:
    """
    compute diagonal operator expectation value from bit string count dictionary

    :param count: count dict for bitstring histogram
    :type count: ct
    :param z: if defaults as None, then ``diagonal_op`` must be set
        a list of qubit that we measure Z op on
    :type z: Optional[Sequence[int]]
    :param diagoal_op: shape [n, 2], explicitly indicate the diagonal op on each qubit
        eg. [1, -1] for z [1, 1] for I, etc.
    :type diagoal_op: Tensor
    :return: the expectation value
    :rtype: float
    """
    if z is None and diagonal_op is None:
        raise ValueError("One of `z` and `diagonal_op` must be set")
    n = len(list(count.keys())[0])
    if z is not None:
        diagonal_op = [[1, -1] if i in z else [1, 1] for i in range(n)]
    r = 0
    shots = 0
    for k, v in count.items():
        cr = 1.0
        for i in range(n):
            cr *= diagonal_op[i][int(k[i])]  # type: ignore
        r += cr * v  # type: ignore
        shots += v
    return r / shots


def merge_count(*counts: ct) -> ct:
    """
    Merge multiple count dictionaries by summing up their counts

    :param counts: Variable number of count dictionaries
    :type counts: ct
    :return: Merged count dictionary
    :rtype: ct

    :Example:

    >>> merge_count({"00": 10, "01": 20}, {"00": 5, "10": 15})
    {'00': 15, '01': 20, '10': 15}
    """
    merged: ct = {}
    for count in counts:
        for k, v in count.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def plot_histogram(data: Any, **kws: Any) -> Any:
    """
    See ``qiskit.visualization.plot_histogram``:
    https://qiskit.org/documentation/stubs/qiskit.visualization.plot_histogram.html

    interesting kw options include: ``number_to_keep`` (int)

    :param data: _description_
    :type data: Any
    :return: _description_
    :rtype: Any
    """
    from qiskit.visualization import plot_histogram

    return plot_histogram(data, **kws)
