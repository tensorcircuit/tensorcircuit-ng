"""
Fermion Gaussian state simulator
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import openfermion
except ModuleNotFoundError:
    pass

from .cons import backend, dtypestr, rdtypestr, get_backend
from .circuit import Circuit
from . import quantum

Tensor = Any


def onehot_matrix(i: int, j: int, N: int) -> Tensor:
    m = np.zeros([N, N])
    m[i, j] = 1
    m = backend.convert_to_tensor(m)
    m = backend.cast(m, dtypestr)
    return m


# TODO(@refraction-ray): efficiency benchmark with jit
# TODO(@refraction-ray): FGS mixed state support?
# TODO(@refraction-ray): overlap?
# TODO(@refraction-ray): fermionic logarithmic negativity


class FGSSimulator:
    r"""
    Fermion Gaussian State (FGS) simulator for efficient simulation of non-interacting fermionic systems.

    This simulator implements methods for:
    1. State initialization and evolution
    2. Correlation matrix calculations
    3. Entanglement measures
    4. Out-of-time-ordered correlators (OTOC)
    5. Post-selection and measurements

    Main references:
    - Gaussian formalism: https://arxiv.org/pdf/2306.16595.pdf
    - Quantum circuits: https://arxiv.org/abs/2209.06945
    - Theory background: https://scipost.org/SciPostPhysLectNotes.54/pdf

    Conventions:
    - Hamiltonian form: (c^dagger, c)H(c, c^\dagger)
    - Correlation form: <(c, c^\dagger)(c^\dagger, c)>
    - Bogoliubov transformation: c' = \alpha^\dagger (c, c^\dagger)

    :Example:

    >>> import tensorcircuit as tc
    >>> # Initialize a 4-site system with sites 0 and 2 occupied
    >>> sim = tc.FGSSimulator(L=4, filled=[0, 2])
    >>> # Apply hopping between sites 0 and 1
    >>> sim.evol_hp(i=0, j=1, chi=1.0)
    >>> # Calculate entanglement entropy for first two sites
    >>> entropy = sim.entropy([2, 3])
    """

    def __init__(
        self,
        L: int,
        filled: Optional[List[int]] = None,
        alpha: Optional[Tensor] = None,
        hc: Optional[Tensor] = None,
        cmatrix: Optional[Tensor] = None,
    ):
        """
        _summary_

        :param L: system size
        :type L: int
        :param filled: the fermion site that is fully occupied, defaults to None
        :type filled: Optional[List[int]], optional
        :param alpha: directly specify the alpha tensor as the input
        :type alpha: Optional[Tensor], optional
        :param hc: the input is given as the ground state of quadratic Hamiltonian ``hc``
        :type hc: Optional[Tensor], optional
        :param cmatrix: only used for debug, defaults to None
        :type cmatrix: Optional[Tensor], optional
        """
        if filled is None:
            filled = []
        self.L = L
        if alpha is None:
            if hc is None:
                self.alpha = self.init_alpha(filled, L)
            else:
                _, _, self.alpha = self.fermion_diagonalization(hc, L)
        else:
            self.alpha = alpha
        self.alpha0 = self.alpha
        self.wtransform = self.wmatrix(L)
        self.cmatrix = cmatrix
        self.otcmatrix: Dict[Tuple[int, int], Tensor] = {}

    @classmethod
    def fermion_diagonalization(
        cls, hc: Tensor, L: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        es, u = backend.eigh(hc)
        es = es[::-1]
        u = u[:, ::-1]
        alpha = u[:, :L]
        return es, u, alpha

    @classmethod
    def fermion_diagonalization_2(
        cls, hc: Tensor, L: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w = cls.wmatrix(L)
        hm = 0.25 * w @ hc @ backend.adjoint(w)
        hm = backend.real((-1.0j) * hm)
        hd, om = backend.schur(hm, output="real")
        # order not kept
        # eps = 1e-10
        # idm = backend.convert_to_tensor(np.array([[1.0, 0], [0, 1.0]]))
        # idm = backend.cast(idm, dtypestr)
        # xm = backend.convert_to_tensor(np.array([[0, 1.0], [1.0, 0]]))
        # xm = backend.cast(xm, dtypestr)
        # for i in range(0, 2 * L, 2):
        #     (backend.sign(hd[i, i + 1] + eps) + 1) / 2 * idm - (
        #         backend.sign(hd[i, i + 1] + eps) - 1
        #     ) / 2 * xm
        # print(hd)

        es = backend.adjoint(w) @ (1.0j * hd) @ w
        u = 0.5 * backend.adjoint(w) @ backend.transpose(om) @ w
        alpha = backend.adjoint(u)[:, :L]
        # c' = u@c
        # e_k (c^\dagger_k c_k - c_k c^\dagger_k)
        return es, u, alpha

    @staticmethod
    def wmatrix(L: int) -> Tensor:
        w = np.zeros([2 * L, 2 * L], dtype=complex)
        for i in range(2 * L):
            if i % 2 == 1:
                w[i, (i - 1) // 2] = 1.0j
                w[i, (i - 1) // 2 + L] = -1.0j
            else:
                w[i, i // 2] = 1
                w[i, i // 2 + L] = 1
        return backend.convert_to_tensor(w)

    @staticmethod
    def init_alpha(filled: List[int], L: int) -> Tensor:
        alpha = np.zeros([2 * L, L])
        for i in range(L):
            if i not in filled:
                alpha[i, i] = 1
            else:
                alpha[i + L, i] = 1
        alpha = backend.convert_to_tensor(alpha)
        alpha = backend.cast(alpha, dtypestr)
        return alpha

    def get_alpha(self) -> Tensor:
        return self.alpha

    def get_cmatrix(self, now_i: bool = True, now_j: bool = True) -> Tensor:
        # otoc in FGS language, see: https://arxiv.org/pdf/1908.03292.pdf
        # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.054205
        # otoc for non=hermitian system is more subtle due to the undefined normalization
        # of operator and not considered here, see: https://arxiv.org/pdf/2305.12054.pdf
        if self.cmatrix is not None and now_i is True and now_j is True:
            return self.cmatrix
        elif now_i is True and now_j is True:
            cmatrix = self.alpha @ backend.adjoint(self.alpha)
            self.cmatrix = cmatrix
            return cmatrix
        elif now_i is True:  # new i old j
            if (1, 0) in self.otcmatrix:
                return self.otcmatrix[(1, 0)]
            cmatrix = self.alpha @ backend.adjoint(self.alpha0)
            self.otcmatrix[(1, 0)] = cmatrix
            return cmatrix
        elif now_j is True:  # old i new j
            if (0, 1) in self.otcmatrix:
                return self.otcmatrix[(0, 1)]
            cmatrix = self.alpha0 @ backend.adjoint(self.alpha)
            self.otcmatrix[(0, 1)] = cmatrix
            return cmatrix
        else:  # initial cmatrix
            if (0, 0) in self.otcmatrix:
                return self.otcmatrix[(0, 0)]
            cmatrix = self.alpha0 @ backend.adjoint(self.alpha0)
            self.otcmatrix[(0, 0)] = cmatrix
            return cmatrix

    def get_reduced_cmatrix(self, subsystems_to_trace_out: List[int]) -> Tensor:
        """
        get reduced correlation matrix by tracing out subsystems

        :param subsystems_to_trace_out: list of sites to be traced out
        :type subsystems_to_trace_out: List[int]
        :return: reduced density matrix
        :rtype: Tensor
        """
        m = self.get_cmatrix()
        if subsystems_to_trace_out is None:
            subsystems_to_trace_out = []
        keep = [i for i in range(self.L) if i not in subsystems_to_trace_out]
        keep += [i + self.L for i in range(self.L) if i not in subsystems_to_trace_out]
        if len(keep) == 0:  # protect from empty keep
            raise ValueError("the full system is traced out, no subsystems to keep")
        keep = backend.convert_to_tensor(keep)

        def slice_(a: Tensor) -> Tensor:
            return backend.gather1d(a, keep)

        slice_ = backend.vmap(slice_)
        m = backend.gather1d(slice_(m), keep)
        return m

    def renyi_entropy(self, n: int, subsystems_to_trace_out: List[int]) -> Tensor:
        """
        compute renyi_entropy of order ``n`` for the fermion state

        :param n: _description_
        :type n: int
        :param subsystems_to_trace_out: system sites to be traced out
        :type subsystems_to_trace_out: List[int]
        :return: _description_
        :rtype: Tensor
        """
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)
        lbd, _ = backend.eigh(m)
        lbd = backend.real(lbd)
        lbd = backend.relu(lbd)
        eps = 1e-6

        entropy = backend.sum(backend.log(lbd**n + (1 - lbd) ** n + eps))
        s = 1 / (2 * (1 - n)) * entropy
        return s

    def charge_moment(
        self,
        alpha: Tensor,
        n: int,
        subsystems_to_trace_out: List[int],
    ) -> Tensor:
        """
        Ref: https://arxiv.org/abs/2302.03330

        :param alpha: to be integrated
        :type alpha: Tensor
        :param n: n-order, Renyi-n
        :type n: int
        :param subsystems_to_trace_out: _description_
        :type subsystems_to_trace_out: List[int]
        :return: _description_
        :rtype: Tensor
        """
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)
        subL = backend.shape_tuple(m)[-1] // 2
        gamma = 2 * m - backend.eye(2 * subL)
        if n == 2:
            eps = 1e-3
        elif n == 3:
            eps = 2e-2
        else:  # >3
            eps = 8e-2
        na = np.concatenate([-np.ones([subL]), np.ones([subL])])
        na = backend.convert_to_tensor(na)
        na = backend.cast(na, dtypestr)
        m = (backend.eye(2 * subL) - gamma) / 2
        for _ in range(n - 1):
            m = m @ (backend.eye(2 * subL) - gamma) / 2
        # m = np.linalg.matrix_power((np.eye(2 * subL) - gamma) / 2, n)
        wprod = backend.eye(2 * subL)
        invm = backend.inv((1 + eps) * backend.eye(2 * subL) - gamma)
        for i in range(n):
            wprod = (
                (((1 + eps) * backend.eye(2 * subL) - gamma) @ (wprod @ invm))
                @ ((backend.eye(2 * subL) + gamma) / 2)
                @ backend.diagflat(
                    backend.exp(1.0j * (alpha[(i + 1) % n] - alpha[i]) * na)
                )
            )
        r = backend.sqrt(backend.det(m + wprod))
        return r

    def renyi_entanglement_asymmetry(
        self,
        n: int,
        subsystems_to_trace_out: List[int],
        batch: int = 100,
        status: Optional[Tensor] = None,
        with_std: bool = False,
    ) -> Tensor:
        """
        Ref: https://arxiv.org/abs/2302.03330

        :param n: _description_
        :type n: int
        :param subsystems_to_trace_out: _description_
        :type subsystems_to_trace_out: List[int]
        :param batch: sample number, defaults 100
        :type batch: int
        :param status: random number for the sample, -pi to pi,
            with shape [batch, n]
        :type status: Optional[Tensor]
        :return: _description_
        :rtype: Tensor
        """
        r = []
        if status is None:
            status = backend.implicit_randu([batch, n], -np.pi, np.pi)
        status = backend.cast(status, dtypestr)
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)
        subL = backend.shape_tuple(m)[-1] // 2
        gamma = 2 * m - backend.eye(2 * subL)
        if n == 2:
            eps = 1e-3
        elif n == 3:
            eps = 2e-2
        else:  # >3
            eps = 8e-2
        na = np.concatenate([-np.ones([subL]), np.ones([subL])])
        na = backend.convert_to_tensor(na)
        na = backend.cast(na, dtypestr)
        m = (backend.eye(2 * subL) - gamma) / 2
        for _ in range(n - 1):
            m = m @ (backend.eye(2 * subL) - gamma) / 2
        # m = np.linalg.matrix_power((np.eye(2 * subL) - gamma) / 2, n)
        invm = backend.inv((1 + eps) * backend.eye(2 * subL) - gamma)
        for i in range(batch):
            alpha = status[i]
            wprod = backend.eye(2 * subL)
            for i in range(n):
                wprod = (
                    (((1 + eps) * backend.eye(2 * subL) - gamma) @ (wprod @ invm))
                    @ ((backend.eye(2 * subL) + gamma) / 2)
                    @ backend.diagflat(
                        backend.exp(1.0j * (alpha[(i + 1) % n] - alpha[i]) * na)
                    )
                )
            r.append(backend.sqrt(backend.det(m + wprod)))
        r = backend.stack(r)
        r_mean = backend.real(backend.mean(r))
        saq = 1 / (1 - n) * backend.log(r_mean)
        if not with_std:
            return saq
        else:
            return saq, backend.abs(1 / (1 - n) * backend.real(backend.std(r)) / saq)

    def entropy(self, subsystems_to_trace_out: Optional[List[int]] = None) -> Tensor:
        """
        compute von Neumann entropy for the fermion state

        :param subsystems_to_trace_out: _description_, defaults to None
        :type subsystems_to_trace_out: Optional[List[int]], optional
        :return: _description_
        :rtype: Tensor
        """
        m = self.get_reduced_cmatrix(subsystems_to_trace_out)  # type: ignore
        lbd, _ = backend.eigh(m)
        lbd = backend.real(lbd)
        lbd = backend.relu(lbd)
        #         lbd /= backend.sum(lbd)
        eps = 1e-6
        entropy = -backend.sum(
            lbd * backend.log(lbd + eps) + (1 - lbd) * backend.log(1 - lbd + eps)
        )
        return entropy / 2

    def evol_hamiltonian(self, h: Tensor) -> None:
        r"""
        Evolve as :math:`e^{-i/2 \hat{h}}`

        :param h: _description_
        :type h: Tensor
        """
        # e^{-i/2 H}
        h = backend.cast(h, dtype=dtypestr)
        self.alpha = backend.expm(-1.0j * h) @ self.alpha
        self.cmatrix = None
        self.otcmatrix = {}

    def evol_ihamiltonian(self, h: Tensor) -> None:
        r"""
        Evolve as :math:`e^{-1/2 \hat{h}}`

        :param h: _description_
        :type h: Tensor
        """
        # e^{-1/2 H}
        h = backend.cast(h, dtype=dtypestr)
        self.alpha = backend.expm(h) @ self.alpha
        self.orthogonal()
        self.cmatrix = None
        self.otcmatrix = {}

    def evol_ghamiltonian(self, h: Tensor) -> None:
        r"""
        Evolve as :math:`e^{-1/2 i \hat{h}}` with h generally non-Hermitian

        :param h: _description_
        :type h: Tensor
        """
        # e^{-1/2 H}
        h = backend.cast(h, dtype=dtypestr)
        self.alpha = backend.expm(-1.0j * backend.adjoint(h)) @ self.alpha
        self.orthogonal()
        self.cmatrix = None
        self.otcmatrix = {}

    def orthogonal(self) -> None:
        q, _ = backend.qr(self.alpha)
        self.alpha = q

    @staticmethod
    def hopping(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        # chi * ci dagger cj + hc.
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, j, 2 * L)
        m += -chi / 2 * onehot_matrix(j + L, i + L, 2 * L)
        m += backend.adjoint(m)
        return m

    def evol_hp(self, i: int, j: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`\chi c_i^\dagger c_j +h.c.`

        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.hopping(chi, i, j, self.L))

    @staticmethod
    def chemical_potential(chi: Tensor, i: int, L: int) -> Tensor:
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, i, 2 * L)
        m += -chi / 2 * onehot_matrix(i + L, i + L, 2 * L)
        return m

    @staticmethod
    def sc_pairing(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        chi = backend.convert_to_tensor(chi)
        chi = backend.cast(chi, dtypestr)
        m = chi / 2 * onehot_matrix(i, j + L, 2 * L)
        m += -chi / 2 * onehot_matrix(j, i + L, 2 * L)
        m += backend.adjoint(m)
        return m

    def evol_sp(self, i: int, j: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_j^\dagger +h.c.`


        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.sc_pairing(chi, i, j, self.L))

    def evol_cp(self, i: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_i`

        :param i: _description_
        :type i: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_hamiltonian(self.chemical_potential(chi, i, self.L))

    def evol_icp(self, i: int, chi: Tensor = 0) -> None:
        r"""
        The evolve Hamiltonian is :math:`chi c_i^\dagger c_i` with :math:`\exp^{-H/2}`

        :param i: _description_
        :type i: int
        :param chi: _description_, defaults to 0
        :type chi: Tensor, optional
        """
        self.evol_ihamiltonian(self.chemical_potential(chi, i, self.L))

    def get_bogoliubov_uv(self) -> Tuple[Tensor, Tensor]:
        return (
            backend.gather1d(
                self.alpha, backend.convert_to_tensor([i for i in range(self.L)])
            ),
            backend.gather1d(
                self.alpha,
                backend.convert_to_tensor([i + self.L for i in range(self.L)]),
            ),
        )

    def get_cmatrix_majorana(self) -> Tensor:
        r"""
        correlation matrix defined in majorana basis
        convention: :math:`gamma_0 = c_0 + c_0^\dagger`
        :math:`gamma_1 = i(c_0 - c_0^\dagger)`

        :return: _description_
        :rtype: Tensor
        """
        c = self.get_cmatrix()
        return self.wtransform @ c @ backend.adjoint(self.wtransform)

    def get_covariance_matrix(self) -> Tensor:
        m = self.get_cmatrix_majorana()
        return -1.0j * (2 * m - backend.eye(self.L * 2))

    def expectation_2body(
        self, i: int, j: int, now_i: bool = True, now_j: bool = True
    ) -> Tensor:
        r"""
        expectation of two fermion terms
        convention: (c, c^\dagger)
        for i>L, c_{i-L}^\dagger is assumed

        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :return: _description_
        :rtype: Tensor
        """
        return self.get_cmatrix(now_i, now_j)[i][(j + self.L) % (2 * self.L)]

    def expectation_4body(self, i: int, j: int, k: int, l: int) -> Tensor:
        r"""
        expectation of four fermion terms using Wick Thm
        convention: (c, c^\dagger)
        for i>L, c_{i-L}^\dagger is assumed

        :param i: _description_
        :type i: int
        :param j: _description_
        :type j: int
        :param k: _description_
        :type k: int
        :param l: _description_
        :type l: int
        :return: _description_
        :rtype: Tensor
        """
        e = (
            self.expectation_2body(i, j) * self.expectation_2body(k, l)
            - self.expectation_2body(i, k) * self.expectation_2body(j, l)
            + self.expectation_2body(i, l) * self.expectation_2body(j, k)
        )
        return e

    def post_select(self, i: int, keep: int = 1) -> None:
        """
        post select (project) the fermion state to occupation eigenstate
        <n_i> = ``keep``

        :param i: _description_
        :type i: int
        :param keep: _description_, defaults to 1
        :type keep: int, optional
        """
        # i is not jittable, keep is jittable
        L = backend.convert_to_tensor(self.L)
        i = backend.convert_to_tensor(i)
        L = backend.cast(L, "int32")
        i = backend.cast(i, "int32")
        keep = backend.convert_to_tensor(keep)
        keep = backend.cast(keep, "int32")
        alpha = self.alpha
        # if keep == 0:
        i = i + L * (1 - keep)
        i0 = backend.argmax(backend.abs(alpha[(i + L) % (2 * L), :]))
        i0 = backend.cast(i0, "int32")
        alpha1 = alpha - backend.reshape(alpha[:, i0], [-1, 1]) @ backend.reshape(
            alpha[(i + L) % (2 * L), :] / alpha[(i + L) % (2 * L), i0], [1, -1]
        )
        mask1 = backend.onehot(i0, alpha.shape[1])
        mask1 = backend.cast(mask1, dtypestr)
        mask0 = backend.ones(alpha.shape[1]) - mask1
        mask12d = backend.tile(mask1[None, :], [alpha.shape[0], 1])
        mask02d = backend.tile(mask0[None, :], [alpha.shape[0], 1])
        alpha1 = mask02d * alpha1 + mask12d * alpha
        r = []
        for j in range(2 * self.L):
            indicator = (
                backend.sign(backend.cast((i - j), rdtypestr) ** 2 - 0.5) + 1
            ) / 2
            # i=j indicator = 0, i!=j indicator = 1
            indicator = backend.cast(indicator, dtypestr)
            r.append(
                backend.ones([self.L]) * indicator
                + backend.ones([self.L]) * mask1 * (1 - indicator)
            )
            # if j != i:
            #     r.append(backend.ones([L]))
            # else:
            #     r.append(backend.ones([L]) * mask1)
        mask2 = backend.stack(r)
        alpha1 = alpha1 * mask2
        r = []
        for j in range(2 * self.L):
            indicator = (
                backend.sign(
                    (backend.cast((i + L) % (2 * L) - j, rdtypestr)) ** 2 - 0.5
                )
                + 1
            ) / 2
            r.append(1 - indicator)
        newcol = backend.stack(r)
        # newcol = np.zeros([2 * self.L])
        # newcol[(i + L) % (2 * L)] = 1
        # newcol = backend.convert_to_tensor(newcol)
        newcol = backend.cast(newcol, dtypestr)
        alpha1 = alpha1 * mask02d + backend.tile(newcol[:, None], [1, self.L]) * mask12d
        q, _ = backend.qr(alpha1)
        self.alpha = q

    def cond_measure(self, ind: int, status: float, with_prob: bool = False) -> Tensor:
        p0 = backend.real(self.get_cmatrix()[ind, ind])
        prob = backend.convert_to_tensor([p0, 1 - p0])
        status = backend.convert_to_tensor(status)
        status = backend.cast(status, rdtypestr)
        eps = 1e-12
        keep = (backend.sign(status - p0 + eps) + 1) / 2
        self.post_select(ind, keep)
        if with_prob is False:
            return keep
        else:
            return keep, prob

    # @classmethod
    # def product(cls, cm1, cm2):
    #     L = cm1.shape([-1])//2
    #     wtransform = cls.wmatrix(L)
    #     gamma1 = -1.0j * (2*wtransform @ cm1 @ backend.adjoint(wtransform)-backend.eye(2*L))
    #     gamma2 = -1.0j * (2*wtransform @ cm2 @ backend.adjoint(wtransform)-backend.eye(2*L))
    #     den = backend.inv(1 + gamma1 @ gamma2)
    #     idm = backend.eye(2 * L)
    #     covm = idm - (idm - gamma2) @ den @ (idm - gamma1)
    #     cm = (1.0j * covm + idm) / 2
    #     cmatrix = backend.adjoint(wtransform) @ cm @ wtransform * 0.25
    #     return cmatrix

    # def product(self, other):
    #     # self@other
    #     gamma1 = self.get_covariance_matrix()
    #     gamma2 = other.get_covariance_matrix()
    #     den = backend.inv(1 + gamma1 @ gamma2)
    #     idm = backend.eye(2 * self.L)
    #     covm = idm - (idm - gamma2) @ den @ (idm - gamma1)
    #     cm = (1.0j * covm + idm) / 2
    #     cmatrix = backend.adjoint(self.wtransform) @ cm @ self.wtransform * 0.25
    #     return type(self)(self.L, cmatrix=cmatrix)

    def overlap(self, other: "FGSSimulator") -> Tensor:
        """
        overlap upto a U(1) phase

        :param other: _description_
        :type other: FGSSimulator
        :return: _description_
        :rtype: _type_
        """
        u, v = self.get_bogoliubov_uv()
        u1, v1 = other.get_bogoliubov_uv()
        return backend.sqrt(
            backend.abs(backend.det(backend.adjoint(u1) @ u + backend.adjoint(v1) @ v))
        )


npb = get_backend("numpy")


class FGSTestSimulator:
    """
    Never use, only for correctness testing
    stick to numpy backend and no jit/ad/vmap is available
    """

    def __init__(
        self,
        L: int,
        filled: Optional[List[int]] = None,
        state: Optional[Tensor] = None,
        hc: Optional[Tensor] = None,
    ):
        if filled is None:
            filled = []
        self.L = L
        if state is not None:
            self.state = state
        elif hc is not None:
            self.state = self.fermion_diagonalization(hc, L)
        else:
            self.state = self.init_state(filled, L)
        self.state0 = self.state

    @staticmethod
    def init_state(filled: List[int], L: int) -> Tensor:
        c = Circuit(L)
        for i in filled:
            c.x(i)  # type: ignore
        return c.state()

    @classmethod
    def fermion_diagonalization(cls, hc: Tensor, L: int) -> Tensor:
        h = cls.get_hmatrix(hc, L)
        _, u = np.linalg.eigh(h)
        return u[:, 0]

    @staticmethod
    def get_hmatrix(hc: Tensor, L: int) -> Tensor:
        hm = np.zeros([2**L, 2**L], dtype=complex)
        for i in range(L):
            for j in range(L):
                op = openfermion.FermionOperator(f"{str(i)}^ {str(j)}")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L, 2 * L):
            for j in range(L):
                op = openfermion.FermionOperator(f"{str(i-L)} {str(j)}")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L):
            for j in range(L, 2 * L):
                op = openfermion.FermionOperator(f"{str(i)}^ {str(j-L)}^")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        for i in range(L, 2 * L):
            for j in range(L, 2 * L):
                op = openfermion.FermionOperator(f"{str(i-L)} {str(j-L)}^")
                hm += (
                    hc[i, j] * openfermion.get_sparse_operator(op, n_qubits=L).todense()
                )

        return hm

    @staticmethod
    def hopping_jw(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(j)}") + np.conj(
            chi
        ) * openfermion.FermionOperator(f"{str(j)}^ {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    @staticmethod
    def chemical_potential_jw(chi: Tensor, i: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    def evol_hamiltonian(self, h: Tensor) -> None:
        self.state = npb.expm(-1 / 2 * 1.0j * h) @ npb.reshape(self.state, [-1, 1])
        self.state = npb.reshape(self.state, [-1])

    def evol_ihamiltonian(self, h: Tensor) -> None:
        self.state = npb.expm(-1 / 2 * h) @ npb.reshape(self.state, [-1, 1])
        self.state = npb.reshape(self.state, [-1])
        self.orthogonal()

    def evol_ghamiltonian(self, h: Tensor) -> None:
        self.state = npb.expm(-1 / 2 * 1.0j * h) @ npb.reshape(self.state, [-1, 1])
        self.state = npb.reshape(self.state, [-1])
        self.orthogonal()

    def evol_hp(self, i: int, j: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.hopping_jw(chi, i, j, self.L))

    def evol_cp(self, i: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.chemical_potential_jw(chi, i, self.L))

    def evol_icp(self, i: int, chi: Tensor = 0) -> None:
        self.evol_ihamiltonian(self.chemical_potential_jw(chi, i, self.L))

    @staticmethod
    def sc_pairing_jw(chi: Tensor, i: int, j: int, L: int) -> Tensor:
        op = chi * openfermion.FermionOperator(f"{str(i)}^ {str(j)}^") + np.conj(
            chi
        ) * openfermion.FermionOperator(f"{str(j)} {str(i)}")
        sop = openfermion.transforms.jordan_wigner(op)
        m = openfermion.get_sparse_operator(sop, n_qubits=L).todense()
        return m

    def evol_sp(self, i: int, j: int, chi: Tensor = 0) -> None:
        self.evol_hamiltonian(self.sc_pairing_jw(chi, i, j, self.L))

    def orthogonal(self) -> None:
        self.state /= backend.norm(self.state)

    def get_ot_cmatrix(self, h: Tensor, now_i: bool = True) -> Tensor:
        alpha1_jw = self.state
        cmatrix = np.zeros([2 * self.L, 2 * self.L], dtype=complex)
        for i in range(self.L):
            for j in range(self.L):
                op1 = openfermion.FermionOperator(f"{str(i)}")
                op2 = openfermion.FermionOperator(f"{str(j)}^")

                m1 = openfermion.get_sparse_operator(op1, n_qubits=self.L).todense()
                m2 = openfermion.get_sparse_operator(op2, n_qubits=self.L).todense()
                eh = npb.expm(-1 / 2 * 1.0j * h)
                eh1 = npb.expm(1 / 2 * 1.0j * h)
                if now_i is True:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ eh1
                            @ m1
                            @ eh
                            @ m2
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
                else:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ m1
                            @ eh1
                            @ m2
                            @ eh
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L):
                op1 = openfermion.FermionOperator(f"{str(i-self.L)}^")
                op2 = openfermion.FermionOperator(f"{str(j)}^")

                m1 = openfermion.get_sparse_operator(op1, n_qubits=self.L).todense()
                m2 = openfermion.get_sparse_operator(op2, n_qubits=self.L).todense()
                eh = npb.expm(-1 / 2 * 1.0j * h)
                eh1 = npb.expm(1 / 2 * 1.0j * h)

                if now_i is True:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ eh1
                            @ m1
                            @ eh
                            @ m2
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
                else:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ m1
                            @ eh1
                            @ m2
                            @ eh
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )

        for i in range(self.L):
            for j in range(self.L, 2 * self.L):
                op1 = openfermion.FermionOperator(f"{str(i)}")
                op2 = openfermion.FermionOperator(f"{str(j-self.L)}")

                m1 = openfermion.get_sparse_operator(op1, n_qubits=self.L).todense()
                m2 = openfermion.get_sparse_operator(op2, n_qubits=self.L).todense()
                eh = npb.expm(-1 / 2 * 1.0j * h)
                eh1 = npb.expm(1 / 2 * 1.0j * h)

                if now_i is True:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ eh1
                            @ m1
                            @ eh
                            @ m2
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
                else:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ m1
                            @ eh1
                            @ m2
                            @ eh
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L, 2 * self.L):
                op1 = openfermion.FermionOperator(f"{str(i-self.L)}^ ")
                op2 = openfermion.FermionOperator(f"{str(j-self.L)}")

                m1 = openfermion.get_sparse_operator(op1, n_qubits=self.L).todense()
                m2 = openfermion.get_sparse_operator(op2, n_qubits=self.L).todense()
                eh = npb.expm(-1 / 2 * 1.0j * h)
                eh1 = npb.expm(1 / 2 * 1.0j * h)

                if now_i is True:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ eh1
                            @ m1
                            @ eh
                            @ m2
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )
                else:
                    cmatrix[i, j] = backend.item(
                        (
                            backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                            @ m1
                            @ eh1
                            @ m2
                            @ eh
                            @ backend.reshape(alpha1_jw, [-1, 1])
                        )[0, 0]
                    )

        return cmatrix

    def get_cmatrix(self) -> Tensor:
        alpha1_jw = self.state
        cmatrix = np.zeros([2 * self.L, 2 * self.L], dtype=complex)
        for i in range(self.L):
            for j in range(self.L):
                op = openfermion.FermionOperator(f"{str(i)} {str(j)}^")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L):
                op = openfermion.FermionOperator(f"{str(i-self.L)}^ {str(j)}^")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L):
            for j in range(self.L, 2 * self.L):
                op = openfermion.FermionOperator(f"{str(i)} {str(j-self.L)}")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        for i in range(self.L, 2 * self.L):
            for j in range(self.L, 2 * self.L):
                op = openfermion.FermionOperator(f"{str(i-self.L)}^ {str(j-self.L)}")
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )
        return cmatrix

    def get_cmatrix_majorana(self) -> Tensor:
        alpha1_jw = self.state
        cmatrix = np.zeros([2 * self.L, 2 * self.L], dtype=complex)
        for i in range(2 * self.L):
            for j in range(2 * self.L):
                op = openfermion.MajoranaOperator((i,)) * openfermion.MajoranaOperator(
                    (j,)
                )
                if j % 2 + i % 2 == 1:
                    op *= -1
                # convention diff in jordan wigner
                # c\dagger = X\pm iY

                op = openfermion.jordan_wigner(op)
                m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
                cmatrix[i, j] = backend.item(
                    (
                        backend.reshape(backend.adjoint(alpha1_jw), [1, -1])
                        @ m
                        @ backend.reshape(alpha1_jw, [-1, 1])
                    )[0, 0]
                )

        return cmatrix

    def expectation_4body(self, i: int, j: int, k: int, l: int) -> Tensor:
        s = ""
        if i < self.L:
            s += str(i) + ""
        else:
            s += str(i - self.L) + "^ "
        if j < self.L:
            s += str(j) + ""
        else:
            s += str(j - self.L) + "^ "
        if k < self.L:
            s += str(k) + ""
        else:
            s += str(k - self.L) + "^ "
        if l < self.L:
            s += str(l) + ""
        else:
            s += str(l - self.L) + "^ "
        op = openfermion.FermionOperator(s)
        m = openfermion.get_sparse_operator(op, n_qubits=self.L).todense()
        return (
            npb.reshape(npb.adjoint(self.state), [1, -1])
            @ m
            @ npb.reshape(self.state, [-1, 1])
        )[0, 0]

    def entropy(self, subsystems_to_trace_out: Optional[List[int]] = None) -> Tensor:
        rm = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)  # type: ignore
        return quantum.entropy(rm)

    def renyi_entropy(self, n: int, subsystems_to_trace_out: List[int]) -> Tensor:
        rm = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)
        return quantum.renyi_entropy(rm, n)

    def charge_moment(
        self, alpha: Tensor, n: int, subsystems_to_trace_out: List[int]
    ) -> Tensor:
        rho = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)
        l = rho.shape[-1]  # type:ignore
        r = np.eye(l)
        L = int(np.log(l) / np.log(2) + 0.001)
        qa = np.diagonal(
            quantum.PauliStringSum2Dense(
                [quantum.xyz2ps({"z": [i]}, L) for i in range(L)]
            )
        )
        qa = qa.reshape([-1])
        for i in range(n):
            r = r @ (rho @ np.diag(np.exp(1.0j * (alpha[(i + 1) % n] - alpha[i]) * qa)))  # type: ignore
        return np.trace(r)

    def renyi_entanglement_asymmetry(
        self, n: int, subsystems_to_trace_out: List[int]
    ) -> Tensor:
        rho = quantum.reduced_density_matrix(self.state, subsystems_to_trace_out)
        l = rho.shape[-1]  # type: ignore
        L = int(np.log(l) / np.log(2) + 0.001)

        qa = np.diagonal(
            quantum.PauliStringSum2Dense(
                [quantum.xyz2ps({"z": [i]}, L) for i in range(L)]
            )
        )
        qa = qa.reshape([-1])
        mask = np.zeros([2**L, 2**L])
        for i in range(2**L):
            for j in range(2**L):
                if qa[i] == qa[j]:
                    mask[i, j] = 1
        return quantum.renyi_entropy(mask * rho, n)

    def overlap(self, other: "FGSTestSimulator") -> Tensor:
        return backend.tensordot(backend.conj(self.state), other.state, 1)

    def get_dm(self) -> Tensor:
        s = backend.reshape(self.state, [-1, 1])
        return s @ backend.adjoint(s)

    def product(self, other: "FGSTestSimulator") -> Tensor:
        rho1 = self.get_dm()
        rho2 = other.get_dm()
        rho = rho1 @ rho2
        rho /= backend.trace(rho)
        return rho

    def post_select(self, i: int, keep: int = 1) -> None:
        c = Circuit(self.L, inputs=self.state)
        c.post_select(i, keep)
        s = c.state()
        s /= backend.norm(s)
        self.state = s

    def cond_measure(self, ind: int, status: float, with_prob: bool = False) -> Tensor:
        p0 = self.get_cmatrix()[ind, ind]
        prob = [p0, 1 - p0]
        if status < p0:
            self.post_select(ind, 0)
            keep = 0
        else:
            self.post_select(ind, 1)
            keep = 1

        if with_prob is False:
            return keep
        else:
            return keep, prob
