"""Unrestricted binary RBMs trained on finite-shot measurement outcomes."""

import jax
from jax import random
import numpy as np
import optax

from measurements import unpack_bits
import tensorcircuit as tc

AMPLITUDE_STEPS = 200_000
AMPLITUDE_BATCH = 128
AMPLITUDE_RATE = 0.1
PHASE_STEPS = 48
PHASE_RATE = 0.20
FISHER_DIAGONAL = 0.01


def unpack(parameters, n):
    """Use n hidden units and a dense hidden-to-visible weight matrix."""
    return parameters[:n], parameters[n : 2 * n], parameters[2 * n :].reshape(n, n)


def log_weight(parameters, visible):
    """Log unnormalized visible probability after summing out hidden units."""
    K = tc.backend
    bias, hidden, weights = unpack(parameters, visible.shape[-1])
    return visible @ bias + K.sum(
        jax.nn.softplus(visible @ weights.T + hidden), axis=-1
    )


def make_amplitude_trainer():
    """Compile CD-1 and Optax SGD on resampled Z minibatches."""
    K = tc.backend
    optimizer = optax.sgd(AMPLITUDE_RATE)

    def mean_score(visible, bias, weights):
        hidden = K.sigmoid(visible @ weights.T + bias)
        return K.concat(
            [
                K.mean(visible, axis=0),
                K.mean(hidden, axis=0),
                (hidden.T @ visible / AMPLITUDE_BATCH).reshape(-1),
            ]
        )

    def step(data, carry):
        parameters, state, rng = carry
        n = data.shape[-1]
        bias, hidden, weights = unpack(parameters, n)
        rng, data_key, hidden_key, visible_key = random.split(rng, 4)
        observed = data[random.randint(data_key, (AMPLITUDE_BATCH,), 0, data.shape[0])]
        h = random.bernoulli(hidden_key, K.sigmoid(observed @ weights.T + hidden))
        model = random.bernoulli(
            visible_key, K.sigmoid(K.cast(h, tc.rdtypestr) @ weights + bias)
        )
        gradient = mean_score(
            K.cast(model, tc.rdtypestr), hidden, weights
        ) - mean_score(observed, hidden, weights)
        updates, state = optimizer.update(gradient, state)
        return optax.apply_updates(parameters, updates), state, rng

    def train(data, parameters, rng):
        def update(_, carry):
            return step(data, carry)

        return jax.lax.fori_loop(
            0, AMPLITUDE_STEPS, update, (parameters, optimizer.init(parameters), rng)
        )[0]

    return K.jit(train)


def phase_data(streams, n):
    """Group duplicate observations without changing their Fisher weights."""
    K = tc.backend
    h = K.numpy(tc.gates.h().tensor)
    sd = K.numpy(tc.gates.sd().tensor)
    rotations = (np.kron(h, h), np.kron(h, h @ sd))
    terms, coefficients, counts = [], [], []
    phase_differences = []
    for site in range(n - 1):
        moments = []
        for axis, rotation in enumerate(rotations):
            observed, count = np.unique(
                streams[1 + 2 * site + axis], return_counts=True
            )
            bits = unpack_bits(observed, n)
            pair = 2 * bits[:, site] + bits[:, site + 1]
            candidates = np.repeat(bits[:, None, :], 4, axis=1)
            candidates[:, :, site : site + 2] = unpack_bits(np.arange(4), 2)
            terms.append(candidates)
            coefficients.append(rotation[pair])
            counts.append(count)
            # For W, the other sites all being zero isolates this pair's coherence.
            outside = (bits.sum(axis=1) - bits[:, site] - bits[:, site + 1]) != 0
            parity = np.array([1, -1, -1, 1])[pair[~outside]]
            moments.append(np.average(parity, weights=count[~outside]))
        phase_differences.append(np.arctan2(moments[1], moments[0]))
    angles = np.r_[0.0, np.cumsum(phase_differences)]
    return (
        K.cast(K.convert_to_tensor(np.concatenate(terms)), tc.rdtypestr),
        K.convert_to_tensor(np.concatenate(coefficients)),
        K.cast(K.convert_to_tensor(np.concatenate(counts)), tc.rdtypestr),
        K.cast(K.convert_to_tensor(angles), tc.rdtypestr),
    )


def phase_scores(phase, terms, amplitudes):
    """Per-observation gradient of minus log rotated probability."""
    K = tc.backend
    n = terms.shape[-1]
    _, hidden, weights = unpack(phase, n)
    components = amplitudes * K.exp(0.5j * log_weight(phase, terms))
    # d[-log |sum components|^2] = Im(sum components * d log_weight / sum components).
    ratio = K.imag(components / K.sum(components, axis=1)[:, None])
    posterior = K.sigmoid(terms @ weights.T + hidden)
    visible_score = K.einsum("bk,bkv->bv", ratio, terms)
    hidden_score = K.einsum("bk,bkh->bh", ratio, posterior)
    weight_score = K.einsum("bk,bkh,bkv->bhv", ratio, posterior, terms)
    return K.concat(
        [visible_score, hidden_score, weight_score.reshape(len(terms), -1)], axis=1
    )


def phase_step(phase, state, terms, amplitudes, counts, optimizer):
    """Apply a damped empirical natural-gradient step with a Fisher trust radius."""
    K = tc.backend
    scores = phase_scores(phase, terms, amplitudes)
    weighted = scores * (counts / K.sum(counts))[:, None]
    gradient = K.sum(weighted, axis=0)
    fisher = scores.T @ weighted
    system = fisher + FISHER_DIAGONAL * K.eye(len(phase), dtype=tc.rdtypestr)
    direction = jax.scipy.linalg.solve(system, gradient, assume_a="pos")
    length = K.sqrt(jax.numpy.maximum(direction @ fisher @ direction, 0))
    scale = PHASE_RATE / jax.numpy.maximum(length, PHASE_RATE)
    updates, state = optimizer.update(scale * direction, state)
    return optax.apply_updates(phase, updates), state


def fit_phase(amplitude, initial, terms, coefficients, counts, angles):
    """Initialize the visible phase biases from training shots and freeze amplitudes."""
    K = tc.backend
    n = terms.shape[-1]
    singleton = K.eye(n, dtype=tc.rdtypestr)
    hidden_phase = log_weight(initial, singleton) - initial[:n]
    biases = 2 * angles - (hidden_phase - hidden_phase[0])
    initial = K.concat([biases, initial[n:]])
    optimizer = optax.sgd(1.0)

    def train(phase, configurations, rotations, multiplicities, fixed_amplitude):
        weights = log_weight(fixed_amplitude, configurations)
        weights -= K.max(weights, axis=1)[:, None]
        amplitudes = rotations * K.exp(0.5 * weights)

        def step(_, carry):
            return phase_step(
                *carry, configurations, amplitudes, multiplicities, optimizer
            )

        return jax.lax.fori_loop(0, PHASE_STEPS, step, (phase, optimizer.init(phase)))[
            0
        ]

    return K.jit(train)(initial, terms, coefficients, counts, amplitude)
