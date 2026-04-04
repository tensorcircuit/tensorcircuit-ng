"""Evaluation of compiled scalar graphs using exact arithmetic.

Combines tsim's exact_scalar.py and evaluate.py.
"""

import functools
from typing import Literal, overload, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

# ==============================================================================
# Exact Scalar (from tsim/core/exact_scalar.py)
# ==============================================================================

_E4 = jnp.exp(1j * jnp.pi / 4)  # e^(i*pi/4)
_E4D = jnp.exp(-1j * jnp.pi / 4)  # e^(-i*pi/4)


@jax.jit
def _scalar_mul(d1: jax.Array, d2: jax.Array) -> jax.Array:
    """Multiply two exact scalar coefficient arrays."""
    a1, b1, c1, d1_coeff = d1[..., 0], d1[..., 1], d1[..., 2], d1[..., 3]
    a2, b2, c2, d2_coeff = d2[..., 0], d2[..., 1], d2[..., 2], d2[..., 3]

    A = a1 * a2 + b1 * d2_coeff - c1 * c2 + d1_coeff * b2
    B = a1 * b2 + b1 * a2 + c1 * d2_coeff + d1_coeff * c2
    C = a1 * c2 + b1 * b2 + c1 * a2 - d1_coeff * d2_coeff
    D = a1 * d2_coeff - b1 * c2 - c1 * b2 + d1_coeff * a2

    return jnp.stack([A, B, C, D], axis=-1).astype(d1.dtype)


def _scalar_to_complex(data: jax.Array) -> jax.Array:
    """Convert a (N, 4) array of coefficients to a (N,) array of complex numbers."""
    return data[..., 0] + data[..., 1] * _E4 + data[..., 2] * 1j + data[..., 3] * _E4D


class ExactScalarArray(NamedTuple):
    """Exact scalar array for ZX-calculus phase arithmetic using dyadic representation.

    Represents values of the form (c_0 + c_1·ω + c_2·ω² + c_3·ω³) × 2^power
    where ω = e^(iπ/4).
    """

    coeffs: Array
    power: Array

    @classmethod
    def create(cls, coeffs: Array, power: Array | None = None) -> "ExactScalarArray":
        if power is None:
            power = jnp.zeros(coeffs.shape[:-1], dtype=jnp.int32)
        return cls(coeffs, power)

    def __mul__(self, other: "ExactScalarArray") -> "ExactScalarArray":
        new_coeffs = _scalar_mul(self.coeffs, other.coeffs)
        new_power = self.power + other.power
        return ExactScalarArray(new_coeffs, new_power)

    def reduce(self) -> "ExactScalarArray":
        def cond_fun(carry):
            coeffs, _ = carry
            reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(
                coeffs != 0, axis=-1
            )
            return jnp.any(reducible)

        def body_fun(carry):
            coeffs, power = carry
            reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(
                coeffs != 0, axis=-1
            )
            coeffs = jnp.where(reducible[..., None], coeffs // 2, coeffs)
            power = jnp.where(reducible, power + 1, power)
            return coeffs, power

        new_coeffs, new_power = jax.lax.while_loop(
            cond_fun, body_fun, (self.coeffs, self.power)
        )
        return ExactScalarArray(new_coeffs, new_power)

    def sum(self) -> "ExactScalarArray":
        min_power = jnp.min(self.power, keepdims=True, axis=-1)
        pow = (self.power - min_power)[..., None]
        aligned_coeffs = self.coeffs * 2**pow
        summed_coeffs = jnp.sum(aligned_coeffs, axis=-2)
        return ExactScalarArray(summed_coeffs, min_power.squeeze(-1))

    def prod(self, axis: int = -1) -> "ExactScalarArray":
        if axis < 0:
            axis = self.coeffs.ndim + axis

        if self.coeffs.shape[axis] == 0:
            coeffs_shape = self.coeffs.shape[:axis] + self.coeffs.shape[axis + 1 :]
            result_coeffs = jnp.zeros(coeffs_shape, dtype=self.coeffs.dtype)
            result_coeffs = result_coeffs.at[..., 0].set(1)

            power_shape = self.power.shape[:axis] + self.power.shape[axis + 1 :]
            result_power = jnp.zeros(power_shape, dtype=self.power.dtype)

            return ExactScalarArray(result_coeffs, result_power)

        scanned = lax.associative_scan(_scalar_mul, self.coeffs, axis=axis)
        result_coeffs = jnp.take(scanned, indices=-1, axis=axis)
        result_power = jnp.sum(self.power, axis=axis)

        return ExactScalarArray(result_coeffs, result_power)

    def to_complex(self) -> jax.Array:
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale


# ==============================================================================
# Evaluate (from tsim/compile/evaluate.py)
# ==============================================================================

# Pre-computed exact scalars for phase values, for powers of omega = e^(i*pi/4)
_UNIT_PHASES = jnp.array(
    [
        [1, 0, 0, 0],  # omega^0 = 1
        [0, 1, 0, 0],  # omega^1
        [0, 0, 1, 0],  # omega^2 = i
        [0, 0, 0, -1],  # omega^3
        [-1, 0, 0, 0],  # omega^4 = -1
        [0, -1, 0, 0],  # omega^5
        [0, 0, -1, 0],  # omega^6 = -i
        [0, 0, 0, 1],  # omega^7
    ],
    dtype=jnp.int32,
)

# Lookup table for exact scalars (1 + omega^k)
_ONE_PLUS_PHASES = _UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)


def _matmul_gf2(a: Array, b: Array) -> Array:
    G, T, _ = a.shape
    if G * T == 0:
        return jnp.zeros((b.shape[0], G, T), dtype=b.dtype)
    return (b.astype(jnp.float32) @ a.astype(jnp.float32).reshape(G * T, -1).T).reshape(
        -1, G, T
    ).astype(jnp.uint8) % 2


@jax.jit
def evaluate(circuit: Any, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values."""
    # ====================================================================
    # TYPE A: Node Terms (1 + e^(i*alpha))
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    # a_param_bits: (num_graphs, max_a, n_params), param_vals: (batch_size, n_params,)
    rowsum_a = _matmul_gf2(circuit.a_param_bits, param_vals)
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8

    term_vals_a_exact = _ONE_PLUS_PHASES[phase_idx_a]
    a_mask = (
        jnp.arange(circuit.a_const_phases.shape[1])[None, :]
        < circuit.a_num_terms[:, None]
    )
    term_vals_a_exact = jnp.where(a_mask[..., None], term_vals_a_exact, _IDENTITY)

    term_vals_a = ExactScalarArray.create(term_vals_a_exact)
    summands_a = term_vals_a.prod(axis=-2)

    # ====================================================================
    # TYPE B: Half-Pi Terms (e^(i*beta))
    # Padded values are 0, so they don't affect the sum.
    # ====================================================================
    rowsum_b = _matmul_gf2(circuit.b_param_bits, param_vals)
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8

    sum_phases_b = jnp.sum(phase_idx_b, axis=-1) % 8

    summands_b_exact = _UNIT_PHASES[sum_phases_b]
    summands_b = ExactScalarArray.create(summands_b_exact)

    # ====================================================================
    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    # ====================================================================
    rowsum_a_c = (
        circuit.c_const_bits_a + _matmul_gf2(circuit.c_param_bits_a, param_vals)
    ) % 2
    rowsum_b_c = (
        circuit.c_const_bits_b + _matmul_gf2(circuit.c_param_bits_b, param_vals)
    ) % 2

    exponent_c = (rowsum_a_c * rowsum_b_c) % 2
    sum_exponents_c = jnp.sum(exponent_c, axis=-1) % 2

    summands_c_exact = (1 - 2 * sum_exponents_c)[..., None] * jnp.array(
        [1, 0, 0, 0], dtype=jnp.int32
    )
    summands_c = ExactScalarArray.create(summands_c_exact)

    # ====================================================================
    # TYPE D: Phase Pairs (1 + e^a + e^b - e^g)
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    rowsum_a_d = _matmul_gf2(circuit.d_param_bits_a, param_vals)
    rowsum_b_d = _matmul_gf2(circuit.d_param_bits_b, param_vals)

    alpha = (circuit.d_const_alpha + rowsum_a_d * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b_d * 4) % 8
    gamma = (alpha + beta) % 8

    term_vals_d_exact = (
        _IDENTITY + _UNIT_PHASES[alpha] + _UNIT_PHASES[beta] - _UNIT_PHASES[gamma]
    )
    d_mask = (
        jnp.arange(circuit.d_const_alpha.shape[1])[None, :]
        < circuit.d_num_terms[:, None]
    )
    term_vals_d_exact = jnp.where(d_mask[..., None], term_vals_d_exact, _IDENTITY)

    term_vals_d = ExactScalarArray.create(term_vals_d_exact)
    summands_d = term_vals_d.prod(axis=-2)

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================
    static_phases = ExactScalarArray.create(_UNIT_PHASES[circuit.phase_indices])
    float_factor = ExactScalarArray.create(circuit.floatfactor)

    total_summands = functools.reduce(
        lambda a, b: a * b,
        [summands_a, summands_b, summands_c, summands_d, static_phases, float_factor],
    )

    def res_exact():
        ts = ExactScalarArray(
            total_summands.coeffs, total_summands.power + circuit.power2
        )
        ts = ts.reduce()
        return ts.sum().to_complex()

    def res_approx():
        return jnp.sum(
            total_summands.to_complex()
            * circuit.approximate_floatfactors
            * 2.0**circuit.power2,
            axis=-1,
        )

    return lax.cond(circuit.has_approximate_floatfactors, res_approx, res_exact)
