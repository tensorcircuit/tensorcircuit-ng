"""
helper functions for conversions
"""

from typing import Any, Tuple, List

import numpy as np

from .. import gates

Tensor = Any


def get_ps(qo: Any, n: int) -> Tuple[Tensor, Tensor]:
    """
    Get Pauli string array and weights array for a qubit Hamiltonian
    as a sum of Pauli strings defined in openfermion ``QubitOperator``.

    :param qo: ``openfermion.ops.operators.qubit_operator.QubitOperator``
    :type qo: ``openfermion.ops.operators.qubit_operator.QubitOperator``
    :param n: The number of qubits
    :type n: int
    :return: Pauli String array and weights array
    :rtype: Tuple[Tensor, Tensor]
    """
    value = gates.PAULI_CHAR_TO_INDEX
    terms = qo.terms
    res = []
    wts = []
    for key in terms:
        bit = np.zeros(n, dtype=int)
        for i in range(len(key)):
            bit[key[i][0]] = value[key[i][1]]
        w = terms[key]
        res.append(tuple(bit))
        wts.append(w)
    return np.array(res), np.array(wts)


def QUBO_to_Ising(Q: Tensor) -> Tuple[Tensor, List[float], float]:
    """
    Convert the Q matrix into the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.

    :param Q: The n-by-n square and symmetric Q-matrix.
    :return pauli_terms: A list of 0/1 series, where each element represents a Pauli term.
    A value of 1 indicates the presence of a Pauli-Z operator, while a value of 0 indicates its absence.
    :return weights: A list of weights corresponding to each Pauli term.
    :return offset: A float representing the offset term of the Ising Hamiltonian.
    """

    n = Q.shape[0]

    if Q[0].shape[0] != n:
        raise ValueError("Matrix is not a square matrix.")

    offset = np.triu(Q, 0).sum() / 2
    pauli_terms = []
    weights = -np.sum(Q, axis=1) / 2

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(term.tolist())

    quadratic_weights = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(term.tolist())

            weight = Q[i][j] / 2
            quadratic_weights.append(weight)

    weights = np.concatenate((weights, quadratic_weights), axis=None)

    return pauli_terms, weights, offset
