"""
Sample-based kernels for the two Figure 5 tVMC benchmarks.

Equations (1), (6), (9), and (13) of https://arxiv.org/abs/2603.18148
are evaluated using amplitudes and their direct derivatives at nodes.
"""

import tensorcircuit as tc


class SpinProblem:
    """One spin, or the open four-spin square in the Marshall basis."""

    def __init__(self, nsites, delta_j=-2.0):
        self.nsites = nsites
        self.nparams = 4 if nsites == 1 else 10
        self.bonds = ((0, 1), (2, 3), (0, 2), (1, 3))
        self.couplings = tc.backend.convert_to_tensor(
            [1 + delta_j, 1 + delta_j, 1.0, 1.0]
        )
        self.amplitudes = tc.backend.vmap(self.amplitude, vectorized_argnums=1)
        self.derivatives = tc.backend.vmap(self.derivative, vectorized_argnums=1)
        self.local_batch = tc.backend.vmap(self.local_values, vectorized_argnums=1)
        self.blur_batch = tc.backend.vmap(self.blur_one, vectorized_argnums=(0, 1))

    def amplitude(self, theta, spins):
        """Complex amplitudes; RBM has four weights and one hidden bias."""
        K = tc.backend
        size = self.nparams // 2
        z = theta[:size] + 1j * theta[size:]
        if self.nsites == 1:
            return K.where(spins[0] == 1, z[0], z[1])
        return 2.0 * K.cosh(K.sum(z[:4] * spins) + z[4])

    def derivative(self, theta, spins):
        """Direct real-parameter Jacobian, finite even when cosh vanishes."""
        K = tc.backend
        if self.nsites == 1:
            d = K.cast(K.stack([spins[0] == 1, spins[0] == -1]), "complex128")
        else:
            z = theta[:5] + 1j * theta[5:]
            features = K.concat([K.cast(spins, "complex128"), K.ones([1])])
            d = 2.0 * K.sinh(K.sum(z[:4] * spins) + z[4]) * features
        return K.concat([d, 1j * d])

    def connections(self, spins):
        """Padded local connections, matrix elements, and diagonal energy."""
        K = tc.backend
        if self.nsites == 1:
            return -spins[None, :], K.stack([-1j * spins[0]]), 0.0
        rows, masks, products = [], [], []
        for i, j in self.bonds:
            permutation = list(range(4))
            permutation[i], permutation[j] = permutation[j], permutation[i]
            rows.append(K.stack([spins[l] for l in permutation]))
            masks.append(spins[i] != spins[j])
            products.append(spins[i] * spins[j])
        valid = K.cast(K.stack(masks), "float64")
        return (
            K.stack(rows),
            -0.5 * self.couplings * valid,
            K.sum(self.couplings * K.stack(products)) / 4.0,
        )

    def degree(self, spins):
        """Number of distinct nonzero off-diagonal Hamiltonian connections."""
        K = tc.backend
        _, mels, _ = self.connections(spins)
        return K.sum(K.cast(K.abs(mels) > 0, "float64"))

    def local_values(self, theta, spins, q):
        """Return psi, its Jacobian, H psi, and the unnormalized blur density."""
        K = tc.backend
        neighbors, mels, diagonal = self.connections(spins)
        a = self.amplitude(theta, spins)
        neighbor_a = self.amplitudes(theta, neighbors)
        degrees = K.vmap(self.degree)(neighbors)
        valid = K.cast(K.abs(mels) > 0, "float64")
        density = (1.0 - q) * K.abs(a) ** 2 + q * K.sum(
            valid * K.abs(neighbor_a) ** 2 / degrees
        )
        return (
            a,
            self.derivative(theta, spins),
            diagonal * a + K.sum(mels * neighbor_a),
            density,
        )

    def blur_one(self, spins, random_values, q):
        """Apply (1-q) I + q K_off after sampling the Born distribution."""
        K = tc.backend
        neighbors, mels, _ = self.connections(spins)
        valid = K.cast(K.abs(mels) > 0, "float64")
        index = K.argmax(K.cumsum(valid) > random_values[1] * K.sum(valid))
        return K.where(random_values[0] < q, neighbors[index], spins)

    def initial_chains(self, count):
        """Independent chains start in a fixed allowed computational state."""
        row = [1] if self.nsites == 1 else [1, -1, 1, -1]
        return tc.backend.tile(tc.backend.convert_to_tensor([row]), [count, 1])

    def metropolis(self, theta, chains, key, sweeps=4):
        """Symmetric local exchange proposals; each row is an independent chain."""
        K = tc.backend

        def step(carry, _):
            current, rng, p = carry
            rng, draw = K.random_split(rng)
            values = K.stateful_randu(draw, [current.shape[0], 2])
            if self.nsites == 1:
                proposal = K.where(values[:, :1] < 0.5, -current, current)
            else:
                pair = K.cast(values[:, 0] * 4, "int32")
                candidates = K.vmap(lambda x: self.connections(x)[0])(current)
                selector = K.onehot(pair, 4)
                proposal = K.cast(
                    K.sum(candidates * selector[:, :, None], axis=1), "int64"
                )
            proposed_p = K.abs(self.amplitudes(theta, proposal)) ** 2
            accept = values[:, 1] * p < proposed_p
            return (
                K.where(accept[:, None], proposal, current),
                rng,
                K.where(accept, proposed_p, p),
            )

        initial_p = K.abs(self.amplitudes(theta, chains)) ** 2
        chains, key, _ = K.scan(step, K.arange(sweeps), (chains, key, initial_p))
        return chains, key

    def compress_samples(self, samples):
        """
        Losslessly group repeated observed configurations and retain counts.

        Only configurations present in the MC batch are evaluated. Static
        padding repeats the first sampled configuration with zero count; the
        Hilbert basis is never enumerated. This reduces duplicate RBM/local
        evaluations on these deliberately tiny benchmarks.
        """
        K = tc.backend
        powers = K.convert_to_tensor([2**i for i in range(self.nsites - 1, -1, -1)])
        labels = K.sum((1 - samples) // 2 * powers, axis=1)
        labels, counts = K.unique_with_counts(
            labels, size=2 if self.nsites == 1 else 6, fill_value=labels[0]
        )
        spins = 1 - 2 * ((labels[:, None] // powers) % 2)
        return spins, counts

    def estimate(self, theta, chains, key, q):
        """MCMC and postprocessing; no basis enumeration enters this path."""
        K = tc.backend
        chains, key = self.metropolis(theta, chains, key)
        key, blur_key = K.random_split(key)
        samples = self.blur_batch(
            chains, K.stateful_randu(blur_key, [chains.shape[0], 2]), q
        )
        samples, counts = self.compress_samples(samples)
        values = self.local_batch(theta, samples, q)
        s, f, force_rows, weights = K.cond(
            q == 0,
            lambda: standard_moments(*values[:3], counts),
            lambda: reweighted_moments(*values, counts),
        )
        return s, f, force_rows, weights, counts, chains, key


def standard_moments(amplitudes, derivatives, hpsi, counts=None):
    """
    Born-sampled covariance; subtract an anchor before taking the mean.

    The anchor makes constant sampled scores cancel exactly, including when
    all chains occupy the same state. This preserves the genuine zero-rank
    collapse rather than inverting roundoff left by a large reduction.
    """
    K = tc.backend
    if counts is None:
        counts = K.ones([amplitudes.shape[0]], dtype="float64")
    mass = counts / K.sum(counts)
    scores = derivatives / amplitudes[:, None]
    energies = hpsi / amplitudes
    scores = scores - scores[:1]
    energies = energies - energies[0]
    centered = scores - K.sum(mass[:, None] * scores, axis=0)
    residual = energies - K.sum(mass * energies)
    force_rows = K.conj(centered) * residual[:, None]
    s = K.conj(K.transpose(centered)) @ (mass[:, None] * centered)
    return (
        s,
        K.sum(mass[:, None] * force_rows, axis=0),
        force_rows,
        K.ones([amplitudes.shape[0]], dtype="float64"),
    )


def reweighted_moments(amplitudes, derivatives, hpsi, density, counts=None):
    """
    Node-safe SNIS S and F; all products cancel psi before evaluation.

    Let a=psi/sqrt(R), d=partial psi/sqrt(R), h=Hpsi/sqrt(R),
    z=mean(|a|^2), c=d-a*mean(a* d)/z, e=h-a*E. Then
    S=mean(c* c)/z and F=mean(c* e)/z. No division by psi occurs.
    R is unnormalized, so the unknown wave-function norm cancels in z.
    The common finite-sample factor in Eq. (11) cancels in TDVP; neither
    this SNIS estimate nor the regularized parameter update is unbiased.
    """
    K = tc.backend
    if counts is None:
        counts = K.ones([amplitudes.shape[0]], dtype="float64")
    mass = counts / K.sum(counts)
    scale = K.sqrt(density)
    a, d, h = amplitudes / scale, derivatives / scale[:, None], hpsi / scale
    weights = K.abs(a) ** 2
    z = K.sum(mass * weights)
    mean_d = K.sum(mass[:, None] * K.conj(a)[:, None] * d, axis=0) / z
    energy = K.sum(mass * K.conj(a) * h) / z
    centered = d - a[:, None] * mean_d
    residual = h - a * energy
    force_rows = K.conj(centered) * residual[:, None] / z
    s = K.conj(K.transpose(centered)) @ (mass[:, None] * centered) / z
    return s, K.sum(mass[:, None] * force_rows, axis=0), force_rows, weights


def schmitt_velocity(s, f, force_rows, snr_cutoff=2.0, counts=None):
    """Schmitt soft spectral/SNR regularization for real parameter coordinates."""
    K = tc.backend
    if counts is None:
        counts = K.ones([force_rows.shape[0]], dtype="float64")
    nsamples = K.sum(counts)
    metric = K.real(s)
    eigenvalues, vectors = K.eigh((metric + K.transpose(metric)) / 2.0)
    rho = K.transpose(vectors) @ f
    active = eigenvalues > 1e-14 * eigenvalues[-1]
    safe_ev = K.where(active, eigenvalues, 1.0)
    largest = K.where(eigenvalues[-1] > 0, eigenvalues[-1], 1.0)
    spectral = 1.0 / (1.0 + (1e-10 * largest / safe_ev) ** 6)
    projected = force_rows @ vectors
    variance = K.sum(counts[:, None] * K.abs(projected - rho) ** 2, axis=0) / nsamples
    signal = K.abs(rho) ** 6
    noise = (snr_cutoff**2 * variance / nsamples) ** 3
    denominator = K.where(signal + noise > 0, signal + noise, 1.0)
    snr_filter = K.where(snr_cutoff == 0, 1.0, signal / denominator)
    velocity = vectors @ (
        K.cast(active, "float64") * spectral * snr_filter * K.imag(rho) / safe_ev
    )
    return velocity


def inner_product_moments(amplitudes, derivatives, hpsi):
    """Exact amplitude-inner-product reference, including zero amplitudes."""
    K = tc.backend
    norm = K.real(K.sum(K.conj(amplitudes) * amplitudes))
    overlap = K.conj(K.transpose(derivatives)) @ amplitudes / norm
    energy = K.sum(K.conj(amplitudes) * hpsi) / norm
    s = K.conj(K.transpose(derivatives)) @ derivatives / norm
    s = s - overlap[:, None] * K.conj(overlap)[None, :]
    f = K.conj(K.transpose(derivatives)) @ hpsi / norm - overlap * energy
    return s, f
