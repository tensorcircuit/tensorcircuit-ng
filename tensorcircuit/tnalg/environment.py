"""
Dense MPS/MPO environments and effective local operators.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

Array = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EnvironmentState:
    """Fixed-length left/right environment buffers for a prepared MPS sweep.

    It intentionally contains only JAX arrays.  Plans and cache lifetimes stay
    outside this value, so it can be used directly in ``jit`` or ``scan``.
    """

    left: Tuple[Array, ...]
    right: Tuple[Array, ...]

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], Tuple[int, int]]:
        """
        Flatten the environment state into JAX leaves and static metadata.

        :return: Environment leaves and left/right leaf counts.
        :rtype: Tuple[Tuple[Array, ...], Tuple[int, int]]
        """
        return self.left + self.right, (len(self.left), len(self.right))

    @classmethod
    def tree_unflatten(
        cls, aux_data: Tuple[int, int], leaves: Tuple[Array, ...]
    ) -> "EnvironmentState":
        """
        Reconstruct an environment state from flattened JAX leaves.

        :param aux_data: Left and right environment leaf counts.
        :type aux_data: Tuple[int, int]
        :param leaves: Flattened environment leaves.
        :type leaves: Tuple[Array, ...]
        :return: Reconstructed environment state.
        :rtype: EnvironmentState
        """
        left_count, right_count = aux_data
        if len(leaves) != left_count + right_count:
            raise ValueError("environment leaves do not match its fixed structure")
        return cls(tuple(leaves[:left_count]), tuple(leaves[left_count:]))


def boundary_environment(dtype: Any) -> Array:
    """
    Return the open-boundary bra-MPO-ket environment.

    :param dtype: Dtype of the boundary tensor.
    :type dtype: Any
    :return: The scalar open-boundary environment tensor.
    :rtype: Array
    """
    return jnp.ones((1, 1, 1), dtype=dtype)


def update_left(environment: Array, mps_tensor: Array, mpo_tensor: Array) -> Array:
    """
    Contract one site into a left environment.

    :param environment: Left environment before the site.
    :type environment: Array
    :param mps_tensor: MPS tensor at the site.
    :type mps_tensor: Array
    :param mpo_tensor: MPO tensor at the site.
    :type mpo_tensor: Array
    :return: Updated left environment.
    :rtype: Array
    """
    return jnp.einsum(
        "abc,apd,bpqe,cqf->def",
        environment,
        jnp.conj(mps_tensor),
        mpo_tensor,
        mps_tensor,
    )


def update_right(environment: Array, mps_tensor: Array, mpo_tensor: Array) -> Array:
    """
    Contract one site into a right environment.

    :param environment: Right environment after the site.
    :type environment: Array
    :param mps_tensor: MPS tensor at the site.
    :type mps_tensor: Array
    :param mpo_tensor: MPO tensor at the site.
    :type mpo_tensor: Array
    :return: Updated right environment.
    :rtype: Array
    """
    return jnp.einsum(
        "def,apd,bpqe,cqf->abc",
        environment,
        jnp.conj(mps_tensor),
        mpo_tensor,
        mps_tensor,
    )


def build_right_environments(
    mps_tensors: Tuple[Array, ...], mpo_tensors: Tuple[Array, ...]
) -> Tuple[Array, ...]:
    """
    Return ``R_i`` for every cut, including the right boundary ``R_N``.

    :param mps_tensors: MPS site tensors in left-to-right order.
    :type mps_tensors: Tuple[Array, ...]
    :param mpo_tensors: MPO site tensors in left-to-right order.
    :type mpo_tensors: Tuple[Array, ...]
    :return: Right environments indexed by MPS cut.
    :rtype: Tuple[Array, ...]
    """
    right = [boundary_environment(jnp.result_type(mps_tensors[0], mpo_tensors[0]))]
    for mps_tensor, mpo_tensor in zip(reversed(mps_tensors), reversed(mpo_tensors)):
        right.append(update_right(right[-1], mps_tensor, mpo_tensor))
    return tuple(reversed(right))


def build_environments(
    mps_tensors: Tuple[Array, ...], mpo_tensors: Tuple[Array, ...]
) -> EnvironmentState:
    """
    Build all fixed-shape left and right environments for one MPS/MPO pair.

    :param mps_tensors: MPS site tensors in left-to-right order.
    :type mps_tensors: Tuple[Array, ...]
    :param mpo_tensors: MPO site tensors in left-to-right order.
    :type mpo_tensors: Tuple[Array, ...]
    :return: Complete left and right environment state.
    :rtype: EnvironmentState
    """
    dtype = jnp.result_type(mps_tensors[0], mpo_tensors[0])
    left = [boundary_environment(dtype)]
    for mps_tensor, mpo_tensor in zip(mps_tensors, mpo_tensors):
        left.append(update_left(left[-1], mps_tensor, mpo_tensor))
    return EnvironmentState(
        tuple(left), build_right_environments(mps_tensors, mpo_tensors)
    )


def local_matvec(left: Array, mpo_tensor: Array, right: Array, vector: Array) -> Array:
    """
    Apply the one-site effective Hamiltonian to a center tensor.

    :param left: Left effective environment.
    :type left: Array
    :param mpo_tensor: Local MPO tensor.
    :type mpo_tensor: Array
    :param right: Right effective environment.
    :type right: Array
    :param vector: Center tensor or vector to which the effective Hamiltonian
        is applied.
    :type vector: Array
    :return: Effective-Hamiltonian action with the center-tensor shape.
    :rtype: Array
    """
    return jnp.einsum("abc,bpqe,def,cqf->apd", left, mpo_tensor, right, vector)
