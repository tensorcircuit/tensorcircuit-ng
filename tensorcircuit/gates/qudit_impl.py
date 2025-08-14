"""
Declarations of single-qubit and two-qubit gates and their corresponding matrix.
"""

import sys
from typing import Any, Callable, Optional, Union

import numpy as np
from sympy import mod_inverse, Mod

import tensornetwork as tn

from ..cons import backend, dtypestr, npdtype
from . import Gate, array_to_tensor


thismodule = sys.modules[__name__]

Tensor = Any
Array = Any
Operator = Any  # QuOperator


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    r = int(n**0.5) + 1
    for i in range(5, r, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def _i_matrix_func(d: int) -> Tensor:
    matrix = np.zeros((d, d), dtype=npdtype)
    for i in range(d):
        matrix[i, i] = 1.0
    return matrix


def _x_matrix_func(d: int) -> Tensor:
    r"""
    X_d\ket{j} = \ket{(j + 1) mod d}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[(j + 1) % d, j] = 1.0
    return matrix


def _z_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    Z_d\ket{j} = \omega^{j}\ket{j}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega**j
    return matrix


def _h_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    H_d\ket{j} = \frac{1}{\sqrt{d}}\sum_{k=0}^{d-1}\omega^{jk}\ket{k}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        for k in range(d):
            matrix[j, k] = omega ** (j * k) / np.sqrt(d)
    return matrix.T


def _s_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    S_d\ket{j} = \omega^{j(j + p_d) / 2}\ket{j}
    """
    _pd = 0 if d % 2 == 0 else 1
    matrix = np.zeros((d, d), dtype=complex)
    for j in range(d):
        phase_exp = (j * (j + _pd)) / 2
        matrix[j, j] = omega**phase_exp
    return matrix


class GateMatrices:
    def __init__(self, d: int):
        assert d > 2
        self.d = d
        self.omega = np.exp(2j * np.pi / self.d)

        # Common single qubit gates as np.ndarray objects
        self._i_matrix = _i_matrix_func(self.d)
        self._x_matrix = _x_matrix_func(self.d)
        self._z_matrix = _z_matrix_func(self.d, self.omega)
        self._h_matrix = _h_matrix_func(self.d, self.omega)

        self._ii_matrix = np.kron(self._i_matrix, self._i_matrix)
        self._xx_matrix = np.kron(self._x_matrix, self._x_matrix)
        self._zz_matrix = np.kron(self._z_matrix, self._z_matrix)

        self._ix_matrix = np.kron(self._i_matrix, self._x_matrix)
        self._iz_matrix = np.kron(self._i_matrix, self._z_matrix)
        self._xi_matrix = np.kron(self._x_matrix, self._i_matrix)
        self._zi_matrix = np.kron(self._z_matrix, self._i_matrix)

        self._xz_matrix = np.kron(self._x_matrix, self._z_matrix)
        self._zx_matrix = np.kron(self._z_matrix, self._x_matrix)


def meta_gate(dim: int) -> None:
    gm = GateMatrices(dim)
    for attr in dir(gm):
        if not (attr.startswith("_") and attr.endswith("_matrix")):
            continue
        gate_name = attr[1:-7]
        m = getattr(gm, attr)

        if getattr(m, "ndim", None) == 2:
            n0, n1 = m.shape
            if n0 != n1:
                raise ValueError(
                    f"{gate_name}: gate matrix must be square, got {m.shape}"
                )

            tmp, k = n0, 0
            while tmp % dim == 0 and tmp > 1:
                tmp //= dim
                k += 1
            if tmp != 1:
                raise ValueError(f"{gate_name}: size {n0} is not a power of dim={dim}")
            if k >= 1:
                m = np.reshape(m, newshape=[dim] * (2 * k))
        else:
            if getattr(m, "ndim", 0) % 2 != 0:
                raise ValueError(
                    f"{gate_name}: tensor order must be even, got {m.ndim}"
                )
            if any(s != dim for s in m.shape):
                raise ValueError(
                    f"{gate_name}: all tensor dims must equal dim={dim}, got {m.shape}"
                )

        gf = GateF(m, gate_name)
        setattr(thismodule, gate_name + "gate", gf)
        setattr(thismodule, gate_name + "_gate", gf)
        setattr(thismodule, gate_name, gf)


def _u8_matrix_func(
    d: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
) -> Tensor:
    if not _is_prime(d):
        raise ValueError(
            f"Dimension d={d} is not prime, U8 gate requires a prime dimension."
        )
    if gamma == 0.0:
        raise ValueError("gamma must be non-zero")

    vks = [0] * d
    if d == 3:
        vks = [0, 1, 8]
    else:
        try:
            inv_12 = mod_inverse(12, d)
        except ValueError:
            raise ValueError(
                f"Inverse of 12 mod {d} does not exist. Choose a prime d that does not divide 12."
            )

        for i in range(1, d):
            a = inv_12 * i * (gamma + i * (6 * z + (2 * i - 3) * gamma)) + eps * i
            vks[i] = Mod(a, d)

    # print(vks)
    sum_vks = Mod(sum(vks), d)
    if sum_vks != 0:
        raise ValueError(
            f"Sum of v_k's is not 0 mod {d}. Got {sum_vks}. Check parameters."
        )

    omega = np.exp(2j * np.pi / d)
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega ** vks[j]
    return matrix


def _cphase_matrix_func(d: int, cv: Optional[int] = None) -> Tensor:
    r"""
    Qudit Controlled-z gate
    \ket{r}\ket{s} \rightarrow \omega^{rs}\ket{r}\ket{s} = \ket{r}Z^r\ket{s}

    This gate is also called SUMZ gate, where Z represents Z_d gate.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       Z_d       0         ...     0      │
     SUMZ_d = │ 0        0       Z_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  Z_d^{d-1} │
              └                                           ─┘
    """
    size = d**2
    omega = np.exp(2j * np.pi / d)
    z_matrix = _z_matrix_func(d=d, omega=omega)

    if cv is None:
        z_pows = [np.eye(d, dtype=npdtype)]
        for _ in range(1, d):
            z_pows.append(z_pows[-1] @ z_matrix)

        matrix = np.zeros((size, size), dtype=npdtype)
        for a in range(d):
            rs = a * d
            matrix[rs : rs + d, rs : rs + d] = z_pows[a]
        return matrix

    if not (0 <= cv < d):
        raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")

    matrix = np.eye(size, dtype=npdtype)
    rs = cv * d
    matrix[rs : rs + d, rs : rs + d] = z_matrix

    return matrix


def _csum_matrix_func(d: int, cv: Optional[int] = None) -> Tensor:
    r"""
    Qudit Controlled-NOT gate
    \ket{r}\ket{s} \rightarrow \ket{r}\ket{r+s} = \ket{r}X^r\ket{s} = \ket{r}\ket{(r+s) mod d}

    This gate is also called SUMX gate, where X represents X_d gate.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       X_d       0         ...     0      │
     SUMX_d = │ 0        0       X_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  X_d^{d-1} │
              └                                           ─┘
    """
    size = d**2
    x_matrix = _x_matrix_func(d=d)

    if cv is None:
        x_pows = [np.eye(d, dtype=npdtype)]
        for _ in range(1, d):
            x_pows.append(x_pows[-1] @ x_matrix)

        matrix = np.zeros((size, size), dtype=npdtype)
        for a in range(d):
            rs = a * d
            matrix[rs : rs + d, rs : rs + d] = x_pows[a]
        return matrix

    if not (0 <= cv < d):
        raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")
    matrix = np.eye(size, dtype=npdtype)
    rs = cv * d
    matrix[rs : rs + d, rs : rs + d] = x_matrix

    return matrix


class VGateMatrices:
    def __init__(self, d: int):
        assert d > 2
        self.d = d

    def u8_gate(self, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0) -> Gate:
        m = _u8_matrix_func(self.d, gamma, z, eps)
        t = array_to_tensor(m)
        t = backend.cast(t, dtypestr)
        t = backend.reshaped(t, self.d)
        return Gate(t, name="u8")

    def cphase_gate(self, cv: Optional[int] = None) -> Gate:
        r"""
        Qudit controlled-Z gate.

        Args:
            cv (Optional[int]): Control value.
                - None: Apply Z_d^r for any control state |r\rangle.
                - int (0 <= cv < d): Apply Z_d only when the control qudit is in state |cv\rangle.
        """
        m = _cphase_matrix_func(self.d, cv)
        t = array_to_tensor(m)
        t = backend.cast(t, dtypestr)
        t = backend.reshaped(t, self.d)
        return Gate(t, name="cz")

    def csum_gate(self, cv: Optional[int] = None) -> Gate:
        """
        Qudit controlled-X (SUM) gate.

        Args:
            cv (Optional[int]): Control value.
                - None: Apply X_d^r for any control state |r\rangle.
                - int (0 <= cv < d): Apply X_d only when the control qudit is in state |cv\rangle.
        """
        m = _csum_matrix_func(self.d, cv)
        t = array_to_tensor(m)
        t = backend.cast(t, dtypestr)
        t = backend.reshaped(t, self.d)
        return Gate(t, name="cnot")

    def mpo_gate(self, mpo: Operator, name: str = "mpo") -> Operator:
        raise NotImplementedError("MPO gate not implemented.")
        # return mpo

    def any_gate(self, unitary: Tensor, name: str = "any") -> Gate:
        """
        Note one should provide the gate with properly reshaped.

        :param unitary: corresponding gate
        :type unitary: Tensor
        :param name: The name of the gate.
        :type name: str
        :return: the resulted gate
        :rtype: Gate
        """
        # deepcopy roadblocks tf.function, pls take care of the unitary outside
        if isinstance(unitary, Gate):
            unitary.tensor = backend.cast(unitary.tensor, dtypestr)
            return unitary
        unitary = backend.cast(unitary, dtypestr)
        try:
            unitary = backend.reshaped(unitary, self.d)
        except ValueError:
            raise ValueError(
                "The dimension of the unitary must be the same as the input dimension."
            )
        return Gate(unitary, name=name)


def meta_vgate(dim: int) -> None:
    vgm = VGateMatrices(dim)
    for attr in ["csum", "cphase", "u8", "any", "mpo"]:
        gvf = GateVF(getattr(vgm, attr + "_gate"), attr)
        for funcname in (attr, attr + "gate", attr + "_gate"):
            setattr(thismodule, funcname, gvf)


def __rmul__(self: tn.Node, lvalue: Union[float, complex]) -> "Gate":
    newg = Gate(lvalue * self.tensor)
    return newg


tn.Node.__rmul__ = __rmul__


class GateF:
    def __init__(self, m: Tensor, n: Optional[str] = None):
        if n is None:
            n = "unknowngate"
        self.m = m
        self.n = n

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        m1 = array_to_tensor(self.m)
        m1 = backend.cast(m1, dtypestr)
        return Gate(m1, name=self.n)

    def adjoint(self) -> "GateF":
        m = self.__call__()
        shape0 = backend.shape_tuple(m.tensor)
        m0 = backend.reshapem(m.tensor)
        ma = backend.adjoint(m0)
        name = self.n + "d"
        ma = backend.reshape(ma, shape0)
        return GateF(ma, name)

    def ided(self, before: bool = True) -> "GateF":
        raise NotImplementedError("Function is not available for qudits.")

    def controlled(self) -> "GateF":
        raise NotImplementedError("Function is not available for qudits.")

    def ocontrolled(self) -> "GateF":
        raise NotImplementedError("Function is not available for qudits.")

    def __str__(self) -> str:
        return self.n

    __repr__ = __str__


class GateVF(GateF):
    def __init__(
        self,
        f: Callable[..., Gate],
        n: Optional[str] = None,
    ):
        if n is None:
            n = "unknowngate"
        self.f = f
        self.n = n

    def __call__(self, *args: Any, **kws: Any) -> Gate:
        return self.f(*args, **kws)

    def adjoint(self) -> "GateVF":
        def f(*args: Any, **kws: Any) -> Gate:
            m = self.__call__(*args, **kws)
            shape0 = backend.shape_tuple(m.tensor)
            m0 = backend.reshapem(m.tensor)
            ma = backend.adjoint(m0)
            # if np.allclose(m0, ma, atol=1e-5):
            #     name = self.n
            # else:
            name = self.n + "d"
            ma = backend.reshape(ma, shape0)
            return Gate(ma, name)

        return GateVF(f, self.n + "d")


# @partial(arg_alias, alias_dict={"unitary": ["hermitian", "hamiltonian"]})
# def exponential_gate(unitary: Tensor, theta: float, name: str = "none") -> Gate:
#     r"""
#     Exponential gate.
#
#     .. math::
#         \textrm{exp}(U) = e^{-j \theta U}
#
#     :param unitary: input unitary :math:`U`
#     :type unitary: Tensor
#     :param theta: angle in radians
#     :type theta: float
#     :param name: suffix of Gate name
#     :return: Exponential Gate
#     :rtype: Gate
#     """
#     theta, unitary = num_to_tensor(theta, unitary)
#     mat = backend.expm(-backend.i() * theta * unitary)
#     dimension = reduce(mul, mat.shape)
#     nolegs = int(np.log(dimension) / np.log(2))
#     mat = backend.reshape(mat, shape=[2 for _ in range(nolegs)])
#     return Gate(mat, name="exp-" + name)


# exp_gate = exponential_gate
# exp = exponential_gate


# @partial(arg_alias, alias_dict={"unitary": ["hermitian", "hamiltonian"]})
# def exponential_gate_unity(
#     unitary: Tensor, theta: float, half: bool = False, name: str = "none"
# ) -> Gate:
#     r"""
#     Faster exponential gate directly implemented based on RHS. Only works when :math:`U^2 = I` is an identity matrix.
#
#     .. math::
#         \textrm{exp}(U) &= e^{-j \theta U} \\
#                 &= \cos(\theta) I - j \sin(\theta) U \\
#
#     :param unitary: input unitary :math:`U`
#     :type unitary: Tensor
#     :param theta: angle in radians
#     :type theta: float
#     :param half: if True, the angel theta is mutiplied by 1/2,
#         defaults to False
#     :type half: bool
#     :param name: suffix of Gate name
#     :type name: str, optional
#     :return: Exponential Gate
#     :rtype: Gate
#     """
#     theta, unitary = num_to_tensor(theta, unitary)
#     size = int(reduce(mul, unitary.shape))
#     n = int(np.log2(size))
#     i = np.eye(2 ** (int(n / 2)))
#     i = i.reshape([2 for _ in range(n)])
#     unitary = backend.reshape(unitary, [2 for _ in range(n)])
#     it = array_to_tensor(i)
#     if half is True:
#         theta = theta / 2.0
#     mat = backend.cos(theta) * it - 1.0j * backend.sin(theta) * unitary
#     return Gate(mat, name="exp1-" + name)
#
#
# exp1_gate = exponential_gate_unity
