"""
Error mitigation methods as objects expanding a circuit into variants and combining
the results of the variants, e.g. for ``variants`` of
:py:func:`tensorcircuit.cloud.wrapper.batch_expectation_ps`

Each method has ``shape``, the shape of the nested list of circuits given by
``expand(circuit)``, and ``combine(values, ideal=None)``, reducing the last
``len(shape)`` axes of the results of the variants, with the ideal values of the
variants for the methods learning from them. The random choices are made by the
generator created from ``seed`` on each ``expand``, so that the same circuit is
expanded into the same variants every time, e.g. once run on the device and once
simulated exactly.
"""

from typing import Any, List, Optional, Sequence, Tuple
import logging

import numpy as np

from ...cons import backend
from .qem_methods import rc_circuit

logger = logging.getLogger(__name__)
Tensor = Any


def _seed(seed: Optional[int]) -> int:
    return np.random.SeedSequence().entropy if seed is None else seed  # type: ignore


def _clifford_angle(theta: Any) -> bool:
    # a multiple of pi/2, up to the float32 precision
    r = float(np.real(backend.numpy(theta))) / (np.pi / 2)
    return abs(r - round(r)) < 1e-6


class Twirl:
    """
    Pauli twirling of the two-qubit gates, by ``n`` random twirls averaged, which
    turns the coherent errors of the gates into Pauli noise. The Pauli gates are
    to be merged into the single qubit gates around by the compiler of the
    provider, e.g. by qiskit before running on LQCloud.
    """

    def __init__(self, n: int, seed: Optional[int] = None):
        """
        :param n: number of the twirled circuits
        :type n: int
        :param seed: the seed of the twirls, defaults to None, i.e. a random one
        :type seed: Optional[int], optional
        """
        self.n = n
        self.seed = _seed(seed)
        self.shape: Tuple[int, ...] = (n,)

    def expand(self, circuit: Any) -> List[Any]:
        rng = np.random.default_rng(self.seed)
        return [rc_circuit(circuit, rng) for _ in range(self.n)]

    def combine(self, values: Tensor, ideal: Optional[Tensor] = None) -> Tensor:
        return np.mean(np.asarray(values), axis=-1)


def cdr_fit(
    noisy: Tensor, ideal: Tensor, intercept: bool = False, atol: float = 1e-12
) -> Tuple[Tensor, Tensor]:
    """
    Least squares fit of ``noisy = a + b * ideal`` on the last axis.

    Where the ideal values hardly vary, i.e. their mean squared deviation is not
    above ``atol``, ``a`` is not fitted. Where they are all close to zero in the
    same sense, e.g. of a Pauli string vanishing by symmetry, or the fitted ``b``
    is not positive, i.e. no signal is left in the noisy values, as Pauli noise
    only gives ``0 < b <= 1``, the noisy value is kept unscaled, i.e. ``a = 0``
    and ``b = 1``, with a warning.

    :param noisy: the noisy values, the samples on the last axis
    :type noisy: Tensor
    :param ideal: the ideal values of the same shape
    :type ideal: Tensor
    :param intercept: whether fit ``a``, otherwise ``a = 0``, defaults to False
    :type intercept: bool, optional
    :param atol: the mean squared ideal values, or their deviations for ``a``,
        not above which the fit is skipped, defaults to 1e-12
    :type atol: float, optional
    :return: ``a`` and ``b`` for each of the leading axes
    :rtype: Tuple[Tensor, Tensor]
    """
    y, x = np.asarray(noisy), np.asarray(ideal)
    sxx = np.mean(x * x, -1)
    zero = sxx <= atol
    a = np.zeros(x.shape[:-1])
    b = np.mean(x * y, -1) / np.where(zero, 1.0, sxx)
    if intercept:
        dx = x - np.mean(x, -1, keepdims=True)
        sdd = np.mean(dx * dx, -1)
        varying = sdd > atol
        bi = np.mean(dx * y, -1) / np.where(varying, sdd, 1.0)
        a = np.where(varying, np.mean(y, -1) - bi * np.mean(x, -1), 0.0)
        b = np.where(varying, bi, b)
    unscaled = zero | (b <= 0)
    if unscaled.any():
        logger.warning(
            "no signal is fitted at %s, i.e. the ideal values of the training "
            "circuits are all close to zero or the slope is not positive, where the "
            "noisy values are kept unscaled",
            np.argwhere(unscaled).tolist(),
        )
    return np.where(unscaled, 0.0, a), np.where(unscaled, 1.0, b)


class CDR:
    """
    Clifford data regression like rescaling: the ``k`` training circuits are the
    circuit with a Gaussian perturbation added to the angle of each ``rz`` gate,
    the only non-Clifford gates of a circuit compiled by
    :py:class:`tensorcircuit.compiler.DefaultCompiler`, except those at the
    multiples of pi/2, so that they run the same gates as the circuit after the
    single qubit gates are merged by the compiler of the provider. ``noisy = a + b
    * ideal`` is fitted on the training
    circuits, and the circuit value is mitigated as ``(noisy - a) / b``, for each
    of the other axes, e.g. for each Pauli string. The fit holds for Pauli noise,
    e.g. after :py:class:`Twirl`, and the training circuits around the circuit,
    e.g. ``sigma`` of 0.1 to 0.3, are better than the Clifford ones, whose ideal
    Pauli expectations are mostly zero. A value whose ideal values hardly vary on
    the training circuits is fitted without ``a``, and kept unscaled if they are
    all close to zero or no signal is left, see :py:func:`cdr_fit`.

    The variants are the circuit and the training circuits, whose ideal values are
    needed by :py:meth:`combine`.
    """

    def __init__(
        self,
        k: int,
        sigma: float = 0.2,
        seed: Optional[int] = None,
        intercept: bool = False,
    ):
        """
        :param k: number of the training circuits
        :type k: int
        :param sigma: the standard deviation of the ``rz`` angle perturbations,
            defaults to 0.2
        :type sigma: float, optional
        :param seed: the seed of the perturbations, defaults to None, i.e. a
            random one
        :type seed: Optional[int], optional
        :param intercept: whether fit the intercept ``a``, which needs ``k`` > 1,
            otherwise ``a = 0``, defaults to False
        :type intercept: bool, optional
        """
        if intercept and k < 2:
            raise ValueError("Fitting the intercept needs at least 2 training circuits")
        self.k = k
        self.sigma = sigma
        self.seed = _seed(seed)
        self.intercept = intercept
        self.shape: Tuple[int, ...] = (k + 1,)

    def expand(self, circuit: Any) -> List[Any]:
        # one generator for each training circuit, so that the ``rz`` gates in
        # common, e.g. of the circuits in different measurement bases, are
        # perturbed the same
        rngs = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(self.seed).spawn(self.k)
        ]
        qir = circuit.to_qir(enable_instruction=True)
        cs = [circuit]
        for rng in rngs:
            perturbed = []
            for d in qir:
                if not d.get("instruction", False) and d["gatef"].n == "rz":
                    theta = d["parameters"]["theta"]
                    delta = rng.normal(scale=self.sigma)
                    # the Clifford angles are kept, by which the compiler of the
                    # provider may save pulses, e.g. merging H rz(pi/2) H into one
                    if not _clifford_angle(theta):
                        d = dict(d, parameters={"theta": theta + delta})
                perturbed.append(d)
            cs.append(type(circuit).from_qir(perturbed, circuit.circuit_param))
        return cs

    def combine(self, values: Tensor, ideal: Optional[Tensor] = None) -> Tensor:
        values = np.asarray(values)
        a, b = cdr_fit(values[..., 1:], np.asarray(ideal)[..., 1:], self.intercept)
        return (values[..., 0] - a) / b


def extrapolate(scales: Sequence[float], values: Tensor, order: int = 1) -> Tensor:
    """
    Extrapolate the values to the zero noise scale by the polynomial fit on the
    last axis, the Richardson extrapolation for ``order = len(scales) - 1``.

    :param scales: the noise scales
    :type scales: Sequence[float]
    :param values: the values at the noise scales on the last axis
    :type values: Tensor
    :param order: the order of the polynomial, defaults to 1
    :type order: int, optional
    :return: the values extrapolated
    :rtype: Tensor
    """
    values = np.asarray(values)
    coef = np.polyfit(scales, values.reshape([-1, len(scales)]).T, order)
    return coef[-1].reshape(values.shape[:-1])


class Fold:
    """
    Zero noise extrapolation by folding each two-qubit gate ``G`` into
    ``G (G^dagger G)^m`` with barriers between, for the noise scales ``2m + 1``.
    Only for the gates whose noise is scaled linearly by folding, e.g. not the CZ
    gates run back to back on some superconducting devices.
    """

    def __init__(self, scales: Sequence[int], order: int = 1):
        """
        :param scales: the noise scales, odd integers
        :type scales: Sequence[int]
        :param order: the order of the polynomial extrapolation, see
            :py:func:`extrapolate`, defaults to 1
        :type order: int, optional
        """
        self.scales = list(scales)
        self.order = order
        self.shape: Tuple[int, ...] = (len(self.scales),)

    def expand(self, circuit: Any) -> List[Any]:
        qir = circuit.to_qir(enable_instruction=True)
        cs = []
        for s in self.scales:
            folded = []
            for d in qir:
                folded.append(d)
                if d.get("instruction", False) or len(d["index"]) != 2:
                    continue
                g = type(circuit)(circuit._nqubits)
                g.append_from_qir([d])
                barrier = {
                    "index": d["index"],
                    "name": "barrier",
                    "gatef": "barrier",
                    "instruction": True,
                }
                folded += [barrier, g.inverse().to_qir()[0], barrier, d] * (
                    (s - 1) // 2
                )
            cs.append(type(circuit).from_qir(folded, circuit.circuit_param))
        return cs

    def combine(self, values: Tensor, ideal: Optional[Tensor] = None) -> Tensor:
        return extrapolate(self.scales, values, self.order)


class Chain:
    """
    Two methods applied in order: each circuit expanded by ``first`` is expanded by
    ``second``, and the results are combined by ``second`` then by ``first``, e.g.
    ``Chain(CDR(2), Twirl(8))`` for the twirled circuit and training circuits.
    """

    def __init__(self, first: Any, second: Any):
        """
        :param first: the outer method
        :type first: Any
        :param second: the inner method
        :type second: Any
        """
        self.first = first
        self.second = second
        self.shape: Tuple[int, ...] = tuple(first.shape) + tuple(second.shape)

    def expand(self, circuit: Any) -> List[Any]:
        return backend.tree_map(  # type: ignore
            self.second.expand, self.first.expand(circuit)
        )

    def combine(self, values: Tensor, ideal: Optional[Tensor] = None) -> Tensor:
        # the ideal values combined with themselves are those of the outer variants
        outer = None if ideal is None else self.second.combine(ideal, ideal)
        return self.first.combine(self.second.combine(values, ideal), outer)
