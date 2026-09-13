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
        k = tc.backend
        size = self.nparams // 2
        z = theta[:size] + 1j * theta[size:]
        if self.nsites == 1:
            return k.where(spins[0] == 1, z[0], z[1])
        return 2.0 * k.cosh(k.sum(z[:4] * spins) + z[4])

    def derivative(self, theta, spins):
        """Direct real-parameter Jacobian, finite even when cosh vanishes."""
        k = tc.backend
        if self.nsites == 1:
            d = k.cast(k.stack([spins[0] == 1, spins[0] == -1]), "complex128")
        else:
            z = theta[:5] + 1j * theta[5:]
            features = k.concat([k.cast(spins, "complex128"), k.ones([1])])
            d = 2.0 * k.sinh(k.sum(z[:4] * spins) + z[4]) * features
        return k.concat([d, 1j * d])

    def connections(self, spins):
        """Padded local connections, matrix elements, and diagonal energy."""
        k = tc.backend
        if self.nsites == 1:
            return -spins[None, :], k.stack([-1j * spins[0]]), 0.0
        rows, masks, products = [], [], []
        for i, j in self.bonds:
            permutation = list(range(4))
            permutation[i], permutation[j] = permutation[j], permutation[i]
            rows.append(k.stack([spins[l] for l in permutation]))
            masks.append(spins[i] != spins[j])
            products.append(spins[i] * spins[j])
        valid = k.cast(k.stack(masks), "float64")
        return (
            k.stack(rows),
            -0.5 * self.couplings * valid,
            k.sum(self.couplings * k.stack(products)) / 4.0,
        )

    def degree(self, spins):
        """Number of distinct nonzero off-diagonal Hamiltonian connections."""
        k = tc.backend
        _, mels, _ = self.connections(spins)
        return k.sum(k.cast(k.abs(mels) > 0, "float64"))

    def local_values(self, theta, spins, q):
        """Return psi, its Jacobian, H psi, and the unnormalized blur density."""
        k = tc.backend
        neighbors, mels, diagonal = self.connections(spins)
        a = self.amplitude(theta, spins)
        neighbor_a = self.amplitudes(theta, neighbors)
        degrees = k.vmap(self.degree)(neighbors)
        valid = k.cast(k.abs(mels) > 0, "float64")
        density = (1.0 - q) * k.abs(a) ** 2 + q * k.sum(
            valid * k.abs(neighbor_a) ** 2 / degrees
        )
        return (
            a,
            self.derivative(theta, spins),
            diagonal * a + k.sum(mels * neighbor_a),
            density,
        )

    def blur_one(self, spins, random_values, q):
        """Apply (1-q) I + q K_off after sampling the Born distribution."""
        k = tc.backend
        neighbors, mels, _ = self.connections(spins)
        valid = k.cast(k.abs(mels) > 0, "float64")
        index = k.argmax(k.cumsum(valid) > random_values[1] * k.sum(valid))
        return k.where(random_values[0] < q, neighbors[index], spins)

    def initial_chains(self, count):
        """Independent chains start in a fixed allowed computational state."""
        row = [1] if self.nsites == 1 else [1, -1, 1, -1]
        return tc.backend.tile(tc.backend.convert_to_tensor([row]), [count, 1])

    def metropolis(self, theta, chains, key, sweeps=4):
        """Symmetric local exchange proposals; each row is an independent chain."""
        k = tc.backend

        def step(carry, _):
            current, rng, p = carry
            rng, draw = k.random_split(rng)
            values = k.stateful_randu(draw, [current.shape[0], 2])
            if self.nsites == 1:
                proposal = k.where(values[:, :1] < 0.5, -current, current)
            else:
                pair = k.cast(values[:, 0] * 4, "int32")
                candidates = k.vmap(lambda x: self.connections(x)[0])(current)
                selector = k.onehot(pair, 4)
                proposal = k.cast(
                    k.sum(candidates * selector[:, :, None], axis=1), "int64"
                )
            proposed_p = k.abs(self.amplitudes(theta, proposal)) ** 2
            accept = values[:, 1] * p < proposed_p
            return (
                k.where(accept[:, None], proposal, current),
                rng,
                k.where(accept, proposed_p, p),
            )

        initial_p = k.abs(self.amplitudes(theta, chains)) ** 2
        chains, key, _ = k.scan(step, k.arange(sweeps), (chains, key, initial_p))
        return chains, key

    def compress_samples(self, samples):
        """
        Losslessly group repeated observed configurations and retain counts.

        Only configurations present in the MC batch are evaluated. Static
        padding repeats the first sampled configuration with zero count; the
        Hilbert basis is never enumerated. This reduces duplicate RBM/local
        evaluations on these deliberately tiny benchmarks.
        """
        k = tc.backend
        powers = k.convert_to_tensor([2**i for i in range(self.nsites - 1, -1, -1)])
        labels = k.sum((1 - samples) // 2 * powers, axis=1)
        labels, counts = k.unique_with_counts(
            labels, size=2 if self.nsites == 1 else 6, fill_value=labels[0]
        )
        spins = 1 - 2 * ((labels[:, None] // powers) % 2)
        return spins, counts

    def estimate(self, theta, chains, key, q):
        """MCMC and postprocessing; no basis enumeration enters this path."""
        k = tc.backend
        chains, key = self.metropolis(theta, chains, key)
        key, blur_key = k.random_split(key)
        samples = self.blur_batch(
            chains, k.stateful_randu(blur_key, [chains.shape[0], 2]), q
        )
        samples, counts = self.compress_samples(samples)
        values = self.local_batch(theta, samples, q)
        s, f, force_rows, weights = k.cond(
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
    k = tc.backend
    if counts is None:
        counts = k.ones([amplitudes.shape[0]], dtype="float64")
    mass = counts / k.sum(counts)
    scores = derivatives / amplitudes[:, None]
    energies = hpsi / amplitudes
    scores = scores - scores[:1]
    energies = energies - energies[0]
    centered = scores - k.sum(mass[:, None] * scores, axis=0)
    residual = energies - k.sum(mass * energies)
    force_rows = k.conj(centered) * residual[:, None]
    s = k.conj(k.transpose(centered)) @ (mass[:, None] * centered)
    return (
        s,
        k.sum(mass[:, None] * force_rows, axis=0),
        force_rows,
        k.ones([amplitudes.shape[0]], dtype="float64"),
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
    k = tc.backend
    if counts is None:
        counts = k.ones([amplitudes.shape[0]], dtype="float64")
    mass = counts / k.sum(counts)
    scale = k.sqrt(density)
    a, d, h = amplitudes / scale, derivatives / scale[:, None], hpsi / scale
    weights = k.abs(a) ** 2
    z = k.sum(mass * weights)
    mean_d = k.sum(mass[:, None] * k.conj(a)[:, None] * d, axis=0) / z
    energy = k.sum(mass * k.conj(a) * h) / z
    centered = d - a[:, None] * mean_d
    residual = h - a * energy
    force_rows = k.conj(centered) * residual[:, None] / z
    s = k.conj(k.transpose(centered)) @ (mass[:, None] * centered) / z
    return s, k.sum(mass[:, None] * force_rows, axis=0), force_rows, weights


def schmitt_velocity(s, f, force_rows, snr_cutoff=2.0, counts=None):
    """Schmitt soft spectral/SNR regularization for real parameter coordinates."""
    k = tc.backend
    if counts is None:
        counts = k.ones([force_rows.shape[0]], dtype="float64")
    nsamples = k.sum(counts)
    metric = k.real(s)
    eigenvalues, vectors = k.eigh((metric + k.transpose(metric)) / 2.0)
    rho = k.transpose(vectors) @ f
    active = eigenvalues > 1e-14 * eigenvalues[-1]
    safe_ev = k.where(active, eigenvalues, 1.0)
    largest = k.where(eigenvalues[-1] > 0, eigenvalues[-1], 1.0)
    spectral = 1.0 / (1.0 + (1e-10 * largest / safe_ev) ** 6)
    projected = force_rows @ vectors
    variance = k.sum(counts[:, None] * k.abs(projected - rho) ** 2, axis=0) / nsamples
    signal = k.abs(rho) ** 6
    noise = (snr_cutoff**2 * variance / nsamples) ** 3
    denominator = k.where(signal + noise > 0, signal + noise, 1.0)
    snr_filter = k.where(snr_cutoff == 0, 1.0, signal / denominator)
    velocity = vectors @ (
        k.cast(active, "float64") * spectral * snr_filter * k.imag(rho) / safe_ev
    )
    return velocity


def inner_product_moments(amplitudes, derivatives, hpsi):
    """Exact amplitude-inner-product reference, including zero amplitudes."""
    k = tc.backend
    norm = k.real(k.sum(k.conj(amplitudes) * amplitudes))
    overlap = k.conj(k.transpose(derivatives)) @ amplitudes / norm
    energy = k.sum(k.conj(amplitudes) * hpsi) / norm
    s = k.conj(k.transpose(derivatives)) @ derivatives / norm
    s = s - overlap[:, None] * k.conj(overlap)[None, :]
    f = k.conj(k.transpose(derivatives)) @ hpsi / norm - overlap * energy
    return s, f
