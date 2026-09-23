"""Tests for matrix-function spectral observables."""

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

import tensorcircuit as tc


def _problem():
    energies = np.array([-1.0, 0.2, 1.1])
    hamiltonian = np.diag(energies).astype(np.complex64)
    probes = np.eye(3, dtype=np.complex64) * np.sqrt(3.0)
    return energies, hamiltonian, probes


def _broadened_reference(energies, queries, broadening, lineshape):
    difference = queries[:, None] - energies[None, :]
    if lineshape == "lorentzian":
        return np.sum(
            broadening / (np.pi * (difference**2 + broadening**2)),
            axis=1,
        )
    return np.sum(
        np.exp(-((difference / broadening) ** 2)) / (np.sqrt(np.pi) * broadening),
        axis=1,
    )


def _kpm_reference(energies, queries, config):
    center = (config.bounds[0] + config.bounds[1]) / 2.0
    scale = (config.bounds[1] - config.bounds[0]) / 2.0
    query_x = (queries - center) / scale
    energy_x = (energies - center) / scale
    query_theta = np.arccos(query_x)
    moments = np.array(
        [np.sum(np.cos(n * np.arccos(energy_x))) for n in range(config.order)]
    )
    n = np.arange(config.order, dtype=np.float64)
    if config.kernel == "dirichlet":
        kernel = np.ones(config.order)
    elif config.kernel == "jackson":
        angle = np.pi / config.order
        kernel = (
            (config.order - n) * np.cos(angle * n)
            + np.sin(angle * n) / np.sin(angle) * np.cos(angle)
        ) / config.order
    else:
        kernel = np.sinh(config.lorentz_lambda * (1.0 - n / config.order)) / np.sinh(
            config.lorentz_lambda
        )
    denominator = np.pi * scale * np.sqrt(1.0 - query_x**2)
    coefficients = np.cos(query_theta[:, None] * n[None, :])
    coefficients[:, 1:] *= 2.0
    return (coefficients / denominator[:, None]) @ (kernel * moments)


def test_trace_normalization_and_thermodynamics(npb):
    energies, hamiltonian, probes = _problem()
    config = tc.matrixfunc.KrylovConfig(4)
    logz = tc.spectral.log_partition_function(
        hamiltonian,
        np.array([0.0, 0.5]),
        method=config,
        probes=probes,
    )
    expected = np.array(
        [np.log(3.0), np.log(np.sum(np.exp(-0.5 * energies)))], dtype=np.float32
    )
    np.testing.assert_allclose(logz, expected, atol=2e-5)

    measure = tc.matrixfunc.slq_measure(hamiltonian, probes, config)
    thermal_energy = tc.spectral.thermal_energy(measure, 0.5)
    expected_energy = np.sum(energies * np.exp(-0.5 * energies)) / np.sum(
        np.exp(-0.5 * energies)
    )
    np.testing.assert_allclose(thermal_energy, expected_energy, atol=2e-5)


def test_dos_prepared_forms_and_lineshapes(npb):
    energies, hamiltonian, probes = _problem()
    kpm = tc.matrixfunc.ChebyshevConfig(48, (-1.2, 1.2), kernel="jackson")
    queries = np.array([-0.8, 0.0, 0.8])
    result = tc.spectral.density_of_states(
        hamiltonian,
        queries,
        method=kpm,
        probes=probes,
        normalization="states",
    )
    np.testing.assert_allclose(
        result,
        _kpm_reference(energies, queries, kpm),
        rtol=3e-5,
        atol=3e-5,
    )

    measure = tc.matrixfunc.slq_measure(
        hamiltonian, probes, tc.matrixfunc.KrylovConfig(4)
    )
    for lineshape in ("lorentzian", "gaussian"):
        slq_result = tc.spectral.density_of_states_from_slq(
            measure,
            queries,
            broadening=0.08,
            lineshape=lineshape,
            normalization="probability",
        )
        np.testing.assert_allclose(
            slq_result,
            _broadened_reference(energies, queries, 0.08, lineshape) / len(energies),
            rtol=3e-5,
            atol=3e-5,
        )


def test_thermal_expectation_and_green_function(npb):
    energies, hamiltonian, probes = _problem()
    observable = np.diag([1.0, 2.0, 3.0]).astype(np.complex64)
    estimate = tc.spectral.thermal_expectation(
        hamiltonian,
        observable,
        0.5,
        probes=probes,
        method=tc.matrixfunc.KrylovConfig(4),
    )
    weights = np.exp(-0.5 * energies)
    expected = np.sum(np.diag(observable) * weights) / np.sum(weights)
    np.testing.assert_allclose(estimate, expected, atol=2e-5)

    right = probes[1]
    z = 0.4 + 0.3j
    green = tc.spectral.resolvent_bilinear(
        hamiltonian,
        right,
        z,
        method=tc.matrixfunc.KrylovConfig(4),
    )
    expected_green = np.vdot(right, np.linalg.solve(z * np.eye(3) - hamiltonian, right))
    np.testing.assert_allclose(green, expected_green, atol=2e-5)

    frequencies = np.array([-0.2, 0.0, 0.3])
    ground = energies[0]
    particle = probes[1]
    green_function = tc.spectral.zero_temperature_greens_function(
        hamiltonian,
        frequencies,
        ground_energy=ground,
        particle_right=particle,
        hole_right=probes[2],
        hole_left=probes[2],
        broadening=0.1,
        method=tc.matrixfunc.KrylovConfig(4),
    )
    np.testing.assert_allclose(
        green_function,
        3.0 / (frequencies + ground + 0.1j - energies[1])
        - 3.0 / (-frequencies + ground - 0.1j - energies[2]),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.spectral_function(green_function),
        -np.imag(
            3.0 / (frequencies + ground + 0.1j - energies[1])
            - 3.0 / (-frequencies + ground - 0.1j - energies[2])
        )
        / np.pi,
        atol=2e-5,
    )


def test_krylov_resolvent_without_left_skips_state_materialization(npb, monkeypatch):
    _, hamiltonian, probes = _problem()
    right = probes[1]
    z = 0.4 + 0.3j

    def forbidden(*args, **kwargs):
        raise AssertionError("lanczos_apply must not run for a quadratic resolvent")

    monkeypatch.setattr(tc.matrixfunc, "lanczos_apply", forbidden)
    result = tc.spectral.resolvent_bilinear(
        hamiltonian,
        right,
        z,
        method=tc.matrixfunc.KrylovConfig(4),
    )
    expected = np.vdot(right, np.linalg.solve(z * np.eye(3) - hamiltonian, right))
    np.testing.assert_allclose(result, expected, atol=2e-5)


def test_time_correlation_and_fft_conventions(npb):
    energies, hamiltonian, probes = _problem()
    times = np.array([0.0, 0.1, 0.2])
    correlation = tc.spectral.time_correlation(
        hamiltonian,
        probes[0],
        probes[0],
        times,
        method=tc.matrixfunc.KrylovConfig(4),
    )
    expected_correlation = 3.0 * np.exp(-1.0j * energies[0] * times)
    np.testing.assert_allclose(correlation, expected_correlation, atol=2e-5)
    frequencies, spectrum = tc.spectral.fft_spectrum(
        correlation, times, window="none", convention="positive"
    )
    expected_spectrum = np.fft.fftshift(
        np.conj(np.fft.fft(np.conj(expected_correlation))) * 0.1
    )
    np.testing.assert_allclose(spectrum, expected_spectrum, atol=2e-5)
    expected_frequency = 2.0 * np.pi * np.array([-1.0, 0.0, 1.0]) / 0.3
    np.testing.assert_allclose(frequencies, expected_frequency, atol=2e-5)


def test_fft_default_preserves_zero_time_sum_rule(npb):
    times = np.linspace(0.0, 20.0, 201)
    correlation = np.exp(-1.3j * times)
    frequencies, spectrum = tc.spectral.fft_spectrum(correlation, times)
    spacing = frequencies[1] - frequencies[0]
    np.testing.assert_allclose(
        np.sum(spectrum) * spacing / (2.0 * np.pi),
        correlation[0],
        atol=5e-6,
        rtol=5e-6,
    )


def test_energy_shift_stability_across_thermal_paths(npb):
    hamiltonian = -100.0 * np.eye(3, dtype=np.complex64)
    probes = np.eye(3, dtype=np.complex64) * np.sqrt(3.0)
    krylov = tc.matrixfunc.KrylovConfig(4)
    measure = tc.matrixfunc.slq_measure(hamiltonian, probes, krylov, probe_batch_size=2)
    energy = tc.spectral.thermal_energy(measure, 10.0, energy_shift=-100.0)
    np.testing.assert_allclose(energy, -100.0, atol=2e-3)
    np.testing.assert_allclose(
        tc.spectral.thermal_energy(measure, 10.0, energy_shift=1.0e6),
        energy,
        atol=2e-6,
    )
    chebyshev = tc.matrixfunc.ChebyshevConfig(64, (-101.0, -99.0))
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        hamiltonian, probes, chebyshev, probe_batch_size=2
    )
    cheb_energy = tc.spectral.thermal_energy(moments, 10.0, energy_shift=-100.0)
    np.testing.assert_allclose(cheb_energy, -100.0, atol=2e-2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_scaled_chebyshev_low_temperature_expectation(backend, highp):
    energies = np.array([0.0, 0.4, 1.1, 2.0], dtype=np.float64)
    hamiltonian = np.diag(energies).astype(np.complex128)
    observable_values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    observable = np.diag(observable_values).astype(np.complex128)
    probes = 2.0 * np.eye(4, dtype=np.complex128)
    beta = 200.0
    config = tc.matrixfunc.ChebyshevConfig(48, (-0.5, 2.5), scaling_steps=16)
    estimate = tc.spectral.thermal_expectation(
        hamiltonian,
        observable,
        beta,
        probes=probes,
        method=config,
        energy_shift=energies[0],
    )
    weights = np.exp(-beta * energies)
    expected = np.sum(observable_values * weights) / np.sum(weights)
    np.testing.assert_allclose(estimate, expected, atol=2e-11)


def test_prepared_kpm_bounds_must_match(npb):
    _, hamiltonian, probes = _problem()
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        hamiltonian,
        probes,
        tc.matrixfunc.ChebyshevConfig(16, (-1.2, 1.2)),
    )
    with pytest.raises(ValueError, match="bounds"):
        tc.spectral.density_of_states_from_moments(
            moments,
            0.0,
            config=tc.matrixfunc.ChebyshevConfig(16, (-2.0, 2.0)),
        )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_probe_batching_and_fft_match_unbatched_paths(backend):
    _, hamiltonian, probes = _problem()
    config = tc.matrixfunc.KrylovConfig(4)
    full = tc.matrixfunc.slq_measure(hamiltonian, probes, config)
    chunked = tc.matrixfunc.slq_measure(hamiltonian, probes, config, probe_batch_size=2)
    np.testing.assert_allclose(full.nodes, chunked.nodes, atol=2e-5)
    np.testing.assert_allclose(full.weights, chunked.weights, atol=2e-5)
    np.testing.assert_array_equal(full.active, chunked.active)

    times = np.arange(8, dtype=np.float32) * 0.1
    correlation = np.exp(0.3j * times)
    frequencies, spectrum = tc.spectral.fft_spectrum(
        correlation, times, window="none", convention="positive"
    )
    expected = np.fft.fftshift(np.conj(np.fft.fft(np.conj(correlation))) * 0.1)
    np.testing.assert_allclose(spectrum, expected, atol=2e-5)
    assert frequencies.shape == spectrum.shape


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_fft_accepts_real_and_complex_inputs(backend):
    times = tc.backend.convert_to_tensor(np.arange(8, dtype=np.float32) * 0.1)
    real = tc.backend.convert_to_tensor(np.ones(8, dtype=np.float32))
    complex_values = tc.backend.convert_to_tensor(np.exp(0.3j * np.arange(8)))
    for correlation in (real, complex_values):
        frequencies, spectrum = tc.spectral.fft_spectrum(
            correlation, times, window="none"
        )
        values = np.asarray(tc.backend.numpy(correlation))
        expected = np.fft.fftshift(np.conj(np.fft.fft(np.conj(values))) * 0.1)
        np.testing.assert_allclose(
            np.asarray(tc.backend.numpy(spectrum)), expected, atol=2e-6
        )
        expected_frequencies = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(8, d=0.1))
        np.testing.assert_allclose(
            np.asarray(tc.backend.numpy(frequencies)), expected_frequencies, atol=2e-6
        )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb")])
def test_nonchebyshev_spectral_backend_paths(backend):
    energies = np.array([-0.7, 0.2, 0.9], dtype=np.float32)
    hamiltonian = tc.backend.convert_to_tensor(np.diag(energies).astype(np.complex64))
    probes = tc.backend.convert_to_tensor(np.sqrt(3.0) * np.eye(3, dtype=np.complex64))
    config = tc.matrixfunc.KrylovConfig(4)

    beta = np.array([0.2, 0.5], dtype=np.float32)
    weights = np.exp(-beta[:, None] * energies[None, :])
    expected_logz = np.log(np.sum(weights, axis=1))
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.log_partition_function(
                hamiltonian,
                beta,
                method=config,
                probes=probes,
                probe_batch_size=2,
            )
        ),
        expected_logz,
        atol=3e-5,
    )

    measure = tc.matrixfunc.slq_measure(hamiltonian, probes, config, probe_batch_size=2)
    scalar_beta = 0.4
    scalar_weights = np.exp(-scalar_beta * energies)
    expected_energy = np.sum(scalar_weights * energies) / np.sum(scalar_weights)
    expected_heat = scalar_beta**2 * (
        np.sum(scalar_weights * energies**2) / np.sum(scalar_weights)
        - expected_energy**2
    )
    np.testing.assert_allclose(
        tc.backend.numpy(tc.spectral.thermal_energy(measure, scalar_beta)),
        expected_energy,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        tc.backend.numpy(tc.spectral.heat_capacity(measure, scalar_beta)),
        expected_heat,
        atol=3e-5,
    )

    queries = np.array([-0.3, 0.4], dtype=np.float32)
    broadening = 0.08
    expected_dos = _broadened_reference(energies, queries, broadening, "lorentzian")
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.density_of_states(
                hamiltonian,
                queries,
                method=config,
                probes=probes,
                broadening=broadening,
                normalization="states",
                probe_batch_size=2,
            )
        ),
        expected_dos,
        atol=3e-5,
    )

    right = probes[1]
    z = 0.6 + 0.25j
    expected_resolvent = 3.0 / (z - energies[1])
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.resolvent_bilinear(hamiltonian, right, z, method=config)
        ),
        expected_resolvent,
        atol=3e-5,
    )

    frequencies = np.array([-0.2, 0.3], dtype=np.float32)
    expected_green = 3.0 / (frequencies + energies[0] + 0.1j - energies[1])
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.zero_temperature_greens_function(
                hamiltonian,
                frequencies,
                ground_energy=energies[0],
                particle_right=right,
                broadening=0.1,
                method=config,
            )
        ),
        expected_green,
        atol=3e-5,
    )

    times = np.array([0.0, 0.15, 0.3, 0.45], dtype=np.float32)
    expected_correlation = 3.0 * np.exp(-1.0j * energies[0] * times)
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.time_correlation(
                hamiltonian, probes[0], probes[0], times, method=config
            )
        ),
        expected_correlation,
        atol=3e-5,
    )

    correlation = tc.backend.convert_to_tensor(expected_correlation)
    frequencies_fft, spectrum = tc.spectral.fft_spectrum(
        correlation,
        tc.backend.convert_to_tensor(times),
        window="none",
        convention="positive",
    )
    expected_spectrum = np.fft.fftshift(
        np.conj(np.fft.fft(np.conj(expected_correlation))) * 0.15
    )
    expected_frequencies = (
        2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(times), d=0.15))
    )
    np.testing.assert_allclose(
        tc.backend.numpy(frequencies_fft), expected_frequencies, atol=3e-5
    )
    np.testing.assert_allclose(tc.backend.numpy(spectrum), expected_spectrum, atol=3e-5)

    observable = tc.backend.convert_to_tensor(
        np.diag(np.array([1.0, -2.0, 0.5], dtype=np.float32)).astype(np.complex64)
    )
    observable_weights = np.exp(-scalar_beta * energies)
    expected_observable = np.sum(
        np.diag(np.asarray(tc.backend.numpy(observable))) * observable_weights
    ) / np.sum(observable_weights)
    np.testing.assert_allclose(
        tc.backend.numpy(
            tc.spectral.thermal_expectation(
                hamiltonian,
                observable,
                scalar_beta,
                probes=probes,
                method=config,
                probe_batch_size=2,
            )
        ),
        expected_observable,
        atol=3e-5,
    )


def test_spectral_jax_jit_vmap_and_ad_match_lehmann(jaxb, highp):
    energies = np.array([-0.7, 0.2, 0.9], dtype=np.float64)
    hamiltonian = tc.backend.convert_to_tensor(np.diag(energies).astype(np.complex128))
    probes = tc.backend.convert_to_tensor(np.sqrt(3.0) * np.eye(3, dtype=np.complex128))
    config = tc.matrixfunc.KrylovConfig(4)

    def logz(beta):
        return tc.spectral.log_partition_function(
            hamiltonian,
            beta,
            method=config,
            probes=probes,
        )

    beta = 0.4
    weights = np.exp(-beta * energies)
    expected_logz = np.log(np.sum(weights))
    expected_energy = np.sum(energies * weights) / np.sum(weights)
    beta_tensor = tc.backend.convert_to_tensor(beta, dtype=tc.rdtypestr)
    np.testing.assert_allclose(
        tc.backend.jit(logz)(beta_tensor), expected_logz, atol=2e-10
    )
    np.testing.assert_allclose(
        tc.backend.jit(tc.backend.grad(lambda value: tc.backend.real(logz(value))))(
            beta_tensor
        ),
        -expected_energy,
        atol=2e-9,
    )
    vectorized = tc.backend.vmap(logz)(
        tc.backend.convert_to_tensor(np.array([0.0, beta], dtype=np.float64))
    )
    np.testing.assert_allclose(
        vectorized,
        np.array([np.log(3.0), expected_logz]),
        atol=2e-10,
    )

    def resolvent(query):
        return tc.spectral.resolvent_bilinear(
            hamiltonian, probes[1], query, method=config
        )

    query = tc.backend.convert_to_tensor(0.4 + 0.25j, dtype=tc.dtypestr)
    np.testing.assert_allclose(
        tc.backend.jit(resolvent)(query),
        3.0 / (0.4 + 0.25j - energies[1]),
        atol=2e-9,
    )

    def correlation(time):
        return tc.spectral.time_correlation(
            hamiltonian, probes[0], probes[0], time, method=config
        )

    time = tc.backend.convert_to_tensor(0.3, dtype=tc.rdtypestr)
    np.testing.assert_allclose(
        tc.backend.jit(correlation)(time),
        3.0 * np.exp(-1.0j * energies[0] * 0.3),
        atol=2e-9,
    )


def test_chebyshev_logz_derivative_jit_matches_energy(jaxb, highp):
    energies = np.array([-0.7, 0.2, 0.9], dtype=np.float64)
    hamiltonian = np.diag(energies).astype(np.complex128)
    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex128)
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        hamiltonian,
        probes,
        tc.matrixfunc.ChebyshevConfig(64, (-1.0, 1.0)),
    )

    def logz(beta):
        return tc.spectral.log_partition_function_from_moments(moments, beta)

    gradient = tc.backend.jit(tc.backend.grad(lambda beta: tc.backend.real(logz(beta))))
    for beta in (0.0, 1.0e-6):
        beta_tensor = tc.backend.convert_to_tensor(beta, dtype=tc.rdtypestr)
        weights = np.exp(-beta * energies)
        expected_logz = np.log(np.sum(weights))
        expected_energy = np.sum(weights * energies) / np.sum(weights)
        np.testing.assert_allclose(
            tc.backend.jit(logz)(beta_tensor), expected_logz, atol=2e-10
        )
        np.testing.assert_allclose(gradient(beta_tensor), -expected_energy, atol=2e-10)


def test_numpy_fft_preserves_active_complex64_dtype(npb):
    values = np.ones(8, dtype=np.complex64)
    transformed = tc.backend.fft(values)
    assert np.asarray(transformed).dtype == np.dtype(np.complex64)


def test_slq_dos_masks_inactive_nodes(npb):
    measure = tc.matrixfunc.LanczosMeasure(
        nodes=np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float32),
        weights=np.array([[1.0, 1000.0], [1.0, 1000.0]], dtype=np.float32),
        active=np.array([[True, False], [True, False]]),
        dimension=np.array(2),
    )
    broadening = 0.1
    energy = 0.2
    expected_line = broadening / (np.pi * (energy * energy + broadening * broadening))
    value = tc.spectral.density_of_states_from_slq(
        measure,
        energy,
        broadening=broadening,
        normalization="probability",
    )
    np.testing.assert_allclose(value, expected_line / 2.0, rtol=2e-6)


def test_heat_capacity_uses_joint_probe_covariance(npb):
    energies = np.array([-0.5, 0.2, 1.0, 1.5], dtype=np.float32)
    weights = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    measure = tc.matrixfunc.LanczosMeasure(
        nodes=energies[:, None],
        weights=weights[:, None],
        active=np.ones((4, 1), dtype=bool),
        dimension=tc.backend.convert_to_tensor(4),
    )
    beta = 0.7
    estimate, standard_error = tc.spectral.heat_capacity(measure, beta, with_std=True)
    thermal_weights = weights * np.exp(-beta * (energies - energies.min()))
    samples = np.stack(
        [thermal_weights, thermal_weights * energies, thermal_weights * energies**2]
    )
    means = np.mean(samples, axis=1)
    covariance = np.cov(samples, bias=False)
    gradient = np.array(
        [
            beta**2 * (-means[2] / means[0] ** 2 + 2.0 * means[1] ** 2 / means[0] ** 3),
            beta**2 * (-2.0 * means[1] / means[0] ** 2),
            beta**2 / means[0],
        ]
    )
    expected_value = beta**2 * (means[2] / means[0] - (means[1] / means[0]) ** 2)
    expected_error = np.sqrt(gradient @ covariance @ gradient / len(energies))
    np.testing.assert_allclose(estimate, expected_value, rtol=2e-5)
    np.testing.assert_allclose(standard_error, expected_error, rtol=2e-5)


def test_physics_defaults_match_exact_spectral_references(npb):
    energies, hamiltonian, probes = _problem()
    beta = 0.4
    weights = np.exp(-beta * energies)
    expected_logz = np.log(np.sum(weights))
    expected_z = np.sum(weights)
    np.testing.assert_allclose(
        tc.spectral.log_partition_function(hamiltonian, beta, probes=probes),
        expected_logz,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.partition_function(hamiltonian, beta, probes=probes),
        expected_z,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.free_energy(expected_logz, beta),
        -expected_logz / beta,
        atol=2e-5,
    )

    queries = np.array([-0.6, 0.3], dtype=np.float32)
    expected_dos = _broadened_reference(energies, queries, 0.08, "lorentzian") / 3.0
    np.testing.assert_allclose(
        tc.spectral.density_of_states(
            hamiltonian,
            queries,
            probes=probes,
            broadening=0.08,
        ),
        expected_dos,
        atol=3e-5,
    )

    right = probes[1]
    z = np.array([0.2 + 0.3j, 0.8 + 0.2j], dtype=np.complex64)
    expected_resolvent = 3.0 / (z - energies[1])
    np.testing.assert_allclose(
        tc.spectral.resolvent_bilinear(hamiltonian, right, z),
        expected_resolvent,
        atol=3e-5,
    )
    times = np.array([0.0, 0.15, 0.4], dtype=np.float32)
    np.testing.assert_allclose(
        tc.spectral.time_correlation(hamiltonian, probes[0], probes[0], times),
        3.0 * np.exp(-1.0j * energies[0] * times),
        atol=3e-5,
    )


def test_prepared_estimators_return_values_or_project_style_pairs(npb):
    energies, hamiltonian, probes = _problem()
    config = tc.matrixfunc.ChebyshevConfig(64, (-1.2, 1.2), kernel="jackson")
    moments = tc.matrixfunc.stochastic_chebyshev_moments(hamiltonian, probes, config)
    queries = np.array([-0.5, 0.25], dtype=np.float32)
    value, error = tc.spectral.density_of_states_from_moments(
        moments,
        queries,
        config=config,
        normalization="states",
        with_std=True,
    )
    np.testing.assert_allclose(
        value, _kpm_reference(energies, queries, config), atol=4e-5
    )
    center = (config.bounds[0] + config.bounds[1]) / 2.0
    scale = (config.bounds[1] - config.bounds[0]) / 2.0
    x = (queries - center) / scale
    energy_x = (energies - center) / scale
    n = np.arange(config.order)
    theta = np.arccos(x)
    energy_theta = np.arccos(energy_x)
    denominator = np.pi * scale * np.sqrt(1.0 - x * x)
    coefficients = np.cos(theta[:, None] * n[None, :])
    coefficients[:, 1:] *= 2.0
    kernel = np.asarray(tc.matrixfunc.kernel_weights(config))
    basis = np.cos(n[:, None] * energy_theta[None, :])
    per_probe = 3.0 * (coefficients / denominator[:, None]) @ (kernel[:, None] * basis)
    expected_error = np.std(per_probe, axis=1, ddof=1) / np.sqrt(3.0)
    np.testing.assert_allclose(error, expected_error, atol=4e-5)

    measure = tc.matrixfunc.slq_measure(
        hamiltonian, probes, tc.matrixfunc.KrylovConfig(4)
    )
    energy, energy_error = tc.spectral.thermal_energy(measure, 0.4, with_std=True)
    thermal_weights = np.exp(-0.4 * energies)
    expected_energy = np.sum(energies * thermal_weights) / np.sum(thermal_weights)
    samples = 3.0 * thermal_weights
    np.testing.assert_allclose(energy, expected_energy, atol=2e-5)
    # The ratio standard error includes denominator covariance, not the error
    # of the numerator alone.
    cov = np.cov(samples * energies, samples, ddof=1)[0, 1]
    variance = np.var(samples * energies, ddof=1) / np.mean(samples) ** 2
    variance += (
        np.mean(samples * energies) ** 2
        * np.var(samples, ddof=1)
        / np.mean(samples) ** 4
    )
    variance -= 2.0 * np.mean(samples * energies) * cov / np.mean(samples) ** 3
    np.testing.assert_allclose(energy_error, np.sqrt(variance / 3.0), atol=3e-5)


def test_chebyshev_thermal_and_response_match_lehmann_reference(npb):
    energies = np.array([-0.7, 0.15, 0.9], dtype=np.float32)
    hamiltonian = np.diag(energies).astype(np.complex64)
    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex64)
    config = tc.matrixfunc.ChebyshevConfig(96, (-1.0, 1.0))
    beta = np.array([0.1, 0.5], dtype=np.float32)
    weights = np.exp(-beta[:, None] * energies[None, :])
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        hamiltonian, probes, config, probe_batch_size=2
    )
    logz = tc.spectral.log_partition_function_from_moments(moments, beta)
    energy = tc.spectral.thermal_energy(moments, beta)
    heat = tc.spectral.heat_capacity(moments, beta)
    expected_logz = np.log(np.sum(weights, axis=1))
    expected_energy = np.sum(weights * energies[None, :], axis=1) / np.sum(
        weights, axis=1
    )
    expected_heat = beta**2 * (
        np.sum(weights * energies[None, :] ** 2, axis=1) / np.sum(weights, axis=1)
        - expected_energy**2
    )
    np.testing.assert_allclose(logz, expected_logz, atol=4e-4)
    np.testing.assert_allclose(energy, expected_energy, atol=4e-4)
    np.testing.assert_allclose(heat, expected_heat, atol=4e-4)

    observable = np.diag(np.array([1.0, -2.0, 0.5], dtype=np.float32)).astype(
        np.complex64
    )
    expected_observable = np.sum(
        weights * np.diag(observable)[None, :], axis=1
    ) / np.sum(weights, axis=1)
    np.testing.assert_allclose(
        tc.spectral.thermal_expectation(
            hamiltonian,
            observable,
            beta,
            probes=probes,
            method=config,
            probe_batch_size=2,
        ),
        expected_observable,
        atol=4e-4,
    )

    right = probes[1]
    z = np.array([0.4 + 0.25j, 1.1 + 0.2j], dtype=np.complex64)
    expected_green = 3.0 / (z - energies[1])
    np.testing.assert_allclose(
        tc.spectral.resolvent_bilinear(hamiltonian, right, z, method=config),
        expected_green,
        atol=4e-4,
    )


def test_response_methods_and_fft_windows_match_direct_sums(npb):
    energies, hamiltonian, probes = _problem()
    vector = probes[0] + 0.3 * probes[2]
    times = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    expected = 3.0 * np.exp(-1.0j * energies[0] * times) + 0.27 * np.exp(
        -1.0j * energies[2] * times
    )
    # left=right here gives squared amplitudes 3 and 0.27.
    np.testing.assert_allclose(
        tc.spectral.time_correlation(
            hamiltonian,
            vector,
            vector,
            times,
            method=tc.matrixfunc.TaylorConfig(24, 2),
        ),
        expected,
        atol=3e-5,
    )
    for window in ("none", "hann", "hamming", "blackman"):
        frequencies, spectrum = tc.spectral.fft_spectrum(
            expected,
            times,
            window=window,
            zero_padding=2,
            convention="negative",
        )
        length = 6
        index = np.arange(4)
        if window == "none":
            weights = np.ones(4)
        elif window == "hann":
            weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * index / 3.0))
        elif window == "hamming":
            weights = 0.54 - 0.46 * np.cos(2.0 * np.pi * index / 3.0)
        else:
            phase = 2.0 * np.pi * index / 3.0
            weights = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
        padded = np.concatenate([expected * weights, np.zeros(2, dtype=complex)])
        expected_spectrum = np.fft.fftshift(np.fft.fft(padded) * 0.2)
        expected_frequencies = (
            2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(length, d=0.2))
        )
        np.testing.assert_allclose(spectrum, expected_spectrum, atol=4e-5)
        np.testing.assert_allclose(frequencies, expected_frequencies, atol=4e-5)


def test_spectral_validation_and_probe_distributions(npb):
    with pytest.raises(ValueError):
        tc.spectral.random_trace_probes(1, num_probes=0, dimension=2)
    with pytest.raises(ValueError):
        tc.spectral.random_trace_probes(
            1, num_probes=2, dimension=2, distribution="bad"
        )
    rademacher = tc.spectral.random_trace_probes(
        7, num_probes=8, dimension=4, distribution="rademacher"
    )
    np.testing.assert_allclose(np.abs(rademacher), 1.0, atol=0.0)
    phases = tc.spectral.random_trace_probes(
        7, num_probes=8, dimension=4, distribution="random_phase"
    )
    np.testing.assert_allclose(np.abs(phases), 1.0, atol=2e-6)
    assert not np.allclose(rademacher, phases)

    energies, hamiltonian, probes = _problem()
    with pytest.raises(ValueError):
        tc.spectral.density_of_states(
            hamiltonian,
            0.0,
            probes=probes,
            broadening=-0.1,
        )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states(
            hamiltonian,
            0.0,
            probes=probes,
            method=tc.matrixfunc.ChebyshevConfig(16, (-1.2, 1.2)),
            broadening=0.1,
        )
    with pytest.raises(ValueError):
        tc.spectral.log_partition_function(hamiltonian, -0.1, probes=probes)
    with pytest.raises(ValueError):
        tc.spectral.free_energy(1.0, 0.0)
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(4), np.array([0.0, 0.1, 0.25, 0.3]))
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        tc.spectral.zero_temperature_greens_function(
            hamiltonian,
            0.0,
            ground_energy=energies[0],
            particle_right=probes[0],
            hole_right=probes[1],
            broadening=0.1,
        )


def test_spectral_prepared_scalar_paths_and_validation(npb):
    energies, hamiltonian, probes = _problem()
    krylov = tc.matrixfunc.KrylovConfig(4)
    measure = tc.matrixfunc.slq_measure(hamiltonian, probes, krylov)
    scalar_queries = np.array([0.0], dtype=np.float32)
    expected_lorentzian = (
        _broadened_reference(energies, scalar_queries, 0.1, "lorentzian")[0] / 3.0
    )
    expected_gaussian = (
        _broadened_reference(energies, scalar_queries, 0.1, "gaussian")[0] / 3.0
    )
    np.testing.assert_allclose(
        tc.spectral.density_of_states_from_slq(
            measure, 0.0, broadening=0.1, lineshape="lorentzian"
        ),
        expected_lorentzian,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.density_of_states_from_slq(
            measure, 0.0, broadening=0.1, lineshape="gaussian"
        ),
        expected_gaussian,
        atol=3e-5,
    )
    with pytest.raises(TypeError):
        tc.spectral.density_of_states_from_slq(1, 0.0, broadening=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.spectral.density_of_states_from_slq(
            measure, 0.0, broadening=0.1, lineshape="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states_from_slq(
            measure, 0.0, broadening=0.1, normalization="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states(hamiltonian, 0.0, probes=probes, method=krylov)
    with pytest.raises(TypeError):
        tc.spectral.density_of_states(
            hamiltonian, 0.0, probes=probes, method=object()  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        tc.spectral.density_of_states_from_moments(
            1, 0.0, config=tc.matrixfunc.ChebyshevConfig(4, (-1.2, 1.2))  # type: ignore[arg-type]
        )
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        hamiltonian, probes, tc.matrixfunc.ChebyshevConfig(8, (-1.2, 1.2))
    )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states_from_moments(
            moments, 0.0, config=tc.matrixfunc.ChebyshevConfig(4, (-1.2, 1.2))
        )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states_from_moments(
            moments,
            0.0,
            config=tc.matrixfunc.ChebyshevConfig(8, (-1.2, 1.2)),
            normalization="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        tc.spectral.density_of_states_from_moments(
            moments,
            1.2,
            config=tc.matrixfunc.ChebyshevConfig(8, (-1.2, 1.2)),
        )

    one_measure = tc.matrixfunc.LanczosMeasure(
        nodes=measure.nodes[:1],
        weights=measure.weights[:1],
        active=measure.active[:1],
        dimension=measure.dimension,
    )
    assert np.isfinite(np.asarray(tc.spectral.thermal_energy(one_measure, 0.2)))
    assert np.isfinite(np.asarray(tc.spectral.heat_capacity(one_measure, 0.2)))
    assert np.isfinite(
        np.asarray(tc.spectral.log_partition_function_from_slq(one_measure, 0.2))
    )
    with pytest.raises(ValueError, match="paired samples"):
        tc.spectral.thermal_energy(one_measure, 0.2, with_std=True)
    with pytest.raises(ValueError, match="paired samples"):
        tc.spectral.heat_capacity(one_measure, 0.2, with_std=True)
    with pytest.raises(ValueError, match="at least two probes"):
        tc.spectral.log_partition_function_from_slq(one_measure, 0.2, with_std=True)
    negative_measure = tc.matrixfunc.LanczosMeasure(
        nodes=np.array([[0.0]], dtype=np.float32),
        weights=np.array([[-1.0]], dtype=np.float32),
        active=np.array([[True]]),
        dimension=np.array(1),
    )
    with pytest.raises(ValueError):
        tc.spectral.log_partition_function_from_slq(
            negative_measure, 0.2, probe_scale=1.0
        )
    with pytest.raises(TypeError):
        tc.spectral.thermal_energy(1, 0.2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.spectral.heat_capacity(1, 0.2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.spectral.thermal_energy(measure, -0.2)
    with pytest.raises(ValueError):
        tc.spectral.thermal_energy(measure, 0.2, energy_shift=[0.0, 1.0])


def test_spectral_thermal_pair_branches_and_green_left_vectors(npb):
    energies = np.array([-0.7, 0.2, 0.9], dtype=np.float32)
    hamiltonian = np.diag(energies).astype(np.complex64)
    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex64)
    config = tc.matrixfunc.ChebyshevConfig(64, (-1.0, 1.0))
    moments = tc.matrixfunc.stochastic_chebyshev_moments(hamiltonian, probes, config)
    beta = np.array([0.2, 0.5], dtype=np.float32)
    weights = np.exp(-beta[:, None] * energies[None, :])
    expected_logz = np.log(np.sum(weights, axis=1))
    expected_energy = np.sum(weights * energies[None, :], axis=1) / np.sum(
        weights, axis=1
    )
    logz, logz_error = tc.spectral.log_partition_function_from_moments(
        moments, beta, with_std=True
    )
    thermal, thermal_error = tc.spectral.thermal_energy(moments, beta, with_std=True)
    np.testing.assert_allclose(logz, expected_logz, atol=4e-4)
    np.testing.assert_allclose(thermal, expected_energy, atol=4e-5)
    expected_logz_error = []
    expected_thermal_error = []
    expected_heat_error = []
    for current_beta in beta:
        samples = 3.0 * np.exp(-current_beta * energies)
        numerator = samples * energies
        second = samples * energies**2
        mean_d, mean_n, mean_2 = np.mean(samples), np.mean(numerator), np.mean(second)
        covariance = np.cov(np.stack([samples, numerator, second]), ddof=1)
        expected_logz_error.append(
            np.std(samples, ddof=1) / np.sqrt(3.0) / np.mean(samples)
        )
        ratio_gradient = np.array([-mean_n / mean_d**2, 1.0 / mean_d])
        ratio_covariance = covariance[:2, :2]
        expected_thermal_error.append(
            np.sqrt(ratio_gradient @ ratio_covariance @ ratio_gradient / 3.0)
        )
        gradient = np.array(
            [
                current_beta**2 * (-mean_2 / mean_d**2 + 2.0 * mean_n**2 / mean_d**3),
                current_beta**2 * (-2.0 * mean_n / mean_d**2),
                current_beta**2 / mean_d,
            ]
        )
        expected_heat_error.append(np.sqrt(gradient @ covariance @ gradient / 3.0))
    np.testing.assert_allclose(logz_error, expected_logz_error, atol=3e-5)
    np.testing.assert_allclose(thermal_error, expected_thermal_error, atol=3e-5)
    heat, heat_error = tc.spectral.heat_capacity(
        moments, beta, with_std=True, energy_shift=0.0
    )
    expected_heat = beta**2 * (
        np.sum(weights * energies[None, :] ** 2, axis=1) / np.sum(weights, axis=1)
        - expected_energy**2
    )
    np.testing.assert_allclose(heat, expected_heat, atol=4e-4)
    np.testing.assert_allclose(heat_error, expected_heat_error, atol=3e-5)
    with pytest.raises(ValueError):
        tc.spectral.log_partition_function_from_moments(
            moments, beta, energy_shift=np.array([0.0, 1.0])
        )

    single_vector_moments = tc.matrixfunc.chebyshev_moments(
        np.diag(np.array([-0.4, 0.6], dtype=np.float32)).astype(np.complex64),
        np.array([1.0, 0.2j], dtype=np.complex64),
        tc.matrixfunc.ChebyshevConfig(32, (-1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.log_partition_function_from_moments(single_vector_moments, 0.3)
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.density_of_states_from_moments(
            single_vector_moments,
            0.0,
            config=tc.matrixfunc.ChebyshevConfig(32, (-1.0, 1.0)),
        )
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.thermal_energy(single_vector_moments, 0.3)
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.heat_capacity(single_vector_moments, 0.3)

    green_energies = np.array([-0.4, 0.6], dtype=np.float32)
    green_hamiltonian = np.diag(green_energies).astype(np.complex64)
    green_probes = np.sqrt(2.0) * np.eye(2, dtype=np.complex64)
    right = green_probes[0]
    left = green_probes[1]
    z = 0.4 + 0.2j
    expected_left = 0.0
    expected_left = np.vdot(
        left, np.linalg.solve(z * np.eye(2) - green_hamiltonian, right)
    )
    np.testing.assert_allclose(
        tc.spectral.resolvent_bilinear(
            green_hamiltonian,
            right,
            z,
            left_vector=left,
            method=tc.matrixfunc.KrylovConfig(3),
        ),
        expected_left,
        atol=3e-5,
    )
    with pytest.raises(TypeError):
        tc.spectral.resolvent_bilinear(hamiltonian, right, z, method=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.spectral.zero_temperature_greens_function(
            green_hamiltonian,
            0.1,
            ground_energy=green_energies[0],
            particle_right=right,
            broadening=0.1,
            branch_sign=0,  # type: ignore[arg-type]
        )
    green = tc.spectral.zero_temperature_greens_function(
        green_hamiltonian,
        0.1,
        ground_energy=green_energies[0],
        particle_right=right,
        particle_left=left,
        hole_right=left,
        hole_left=right,
        broadening=0.1,
        branch_sign=-1,
    )
    expected_green = np.vdot(
        left,
        np.linalg.solve((0.1 + 0.1j) * np.eye(2) - green_hamiltonian, right),
    ) + np.vdot(
        right,
        np.linalg.solve((-0.1 - 0.1j) * np.eye(2) - green_hamiltonian, left),
    )
    np.testing.assert_allclose(green, expected_green, atol=3e-5)


def test_spectral_scalar_time_and_fft_validation(npb):
    _, hamiltonian, probes = _problem()
    scalar_time = 0.2
    expected = np.vdot(
        probes[0],
        np.linalg.solve(
            np.eye(3), probes[0] * np.exp(-1.0j * np.diag(hamiltonian) * scalar_time)
        ),
    )
    np.testing.assert_allclose(
        tc.spectral.time_correlation(
            hamiltonian, probes[0], probes[0], scalar_time, energy_shift=0.3
        ),
        expected * np.exp(0.3j * scalar_time),
        atol=3e-5,
    )
    with pytest.raises(ValueError):
        tc.spectral.time_correlation(hamiltonian, probes[0], probes[0], -0.1)
    with pytest.raises(ValueError):
        tc.spectral.thermal_expectation(
            hamiltonian,
            hamiltonian,
            0.2,
            probes=probes,
            probe_batch_size=0,
        )
    one_time = np.array([0.0], dtype=np.float32)
    one_corr = np.array([2.0 + 1.0j], dtype=np.complex64)
    frequencies, spectrum = tc.spectral.fft_spectrum(
        one_corr, one_time, window="hann", convention="negative"
    )
    np.testing.assert_allclose(frequencies, np.array([0.0]))
    np.testing.assert_allclose(spectrum, one_corr)
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(2), np.array([0.0]))
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(2), np.array([0.0, 0.1]), zero_padding=-1)
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(2), np.array([0.0, 0.1]), convention="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(2), np.array([0.0, 0.1]), window="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(0), np.ones(0))
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones(2), np.array([0.0, -0.1]))


def test_spectral_vector_thermal_partition_and_response_branches(npb):
    energies = np.array([-0.3, 0.1, 0.8], dtype=np.float32)
    hamiltonian = np.diag(energies).astype(np.complex64)
    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex64)
    krylov = tc.matrixfunc.KrylovConfig(4)
    measure = tc.matrixfunc.slq_measure(hamiltonian, probes, krylov)
    beta = np.array([0.2, 0.7], dtype=np.float32)
    shift = -0.5
    weights = np.exp(-beta[:, None] * energies[None, :])
    partition = np.sum(weights, axis=1)
    expected_energy = np.sum(weights * energies[None, :], axis=1) / partition
    expected_heat = beta**2 * (
        np.sum(weights * energies[None, :] ** 2, axis=1) / partition
        - expected_energy**2
    )

    np.testing.assert_allclose(
        tc.spectral.log_partition_function_from_slq(measure, beta, energy_shift=shift),
        np.log(partition),
        atol=3e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.thermal_energy(measure, beta, energy_shift=shift),
        expected_energy,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        tc.spectral.heat_capacity(measure, beta, energy_shift=shift),
        expected_heat,
        atol=3e-5,
    )

    chebyshev = tc.matrixfunc.ChebyshevConfig(64, (-1.0, 1.0))
    np.testing.assert_allclose(
        tc.spectral.log_partition_function(
            hamiltonian,
            beta,
            method=chebyshev,
            probes=probes,
            energy_shift=shift,
        ),
        np.log(partition),
        atol=4e-4,
    )

    scalar_beta = 0.4
    scalar_weights = np.exp(-scalar_beta * energies)
    scalar_partition = np.sum(scalar_weights)
    scalar_samples = 3.0 * scalar_weights
    scalar_relative_error = np.sqrt(np.var(scalar_samples, ddof=1) / 3.0) / np.mean(
        scalar_samples
    )
    partition_value, partition_error = tc.spectral.partition_function(
        hamiltonian,
        scalar_beta,
        method=krylov,
        probes=probes,
        with_std=True,
    )
    np.testing.assert_allclose(partition_value, scalar_partition, atol=3e-5)
    np.testing.assert_allclose(
        partition_error,
        scalar_partition * scalar_relative_error,
        atol=3e-5,
    )
    logz_pair = tc.spectral.log_partition_function_from_slq(
        measure, scalar_beta, with_std=True
    )
    free, free_error = tc.spectral.free_energy(logz_pair, scalar_beta, with_std=True)
    np.testing.assert_allclose(free, -np.log(scalar_partition) / scalar_beta, atol=3e-5)
    np.testing.assert_allclose(free_error, logz_pair[1] / scalar_beta, atol=3e-5)

    particle = probes[1]
    frequencies = np.array([-0.2, 0.3], dtype=np.float32)
    expected_particle = 3.0 / (frequencies + energies[0] + 0.1j - energies[1])
    np.testing.assert_allclose(
        tc.spectral.zero_temperature_greens_function(
            hamiltonian,
            frequencies,
            ground_energy=energies[0],
            particle_right=particle,
            broadening=0.1,
            method=krylov,
        ),
        expected_particle,
        atol=3e-5,
    )

    single_vector = np.array([1.0, 0.2j, -0.3], dtype=np.complex64)
    single_moments = tc.matrixfunc.chebyshev_moments(
        hamiltonian,
        single_vector,
        tc.matrixfunc.ChebyshevConfig(48, (-1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.log_partition_function_from_moments(
            single_moments, 0.35, energy_shift=shift
        )
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.thermal_energy(single_moments, 0.35, energy_shift=shift)
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.heat_capacity(single_moments, 0.35, energy_shift=shift)
    with pytest.raises(ValueError, match="trace estimators"):
        tc.spectral.thermal_energy(
            single_moments, 0.35, energy_shift=np.array([0.0, 1.0])
        )

    negative_measure = tc.matrixfunc.LanczosMeasure(
        nodes=np.zeros((2, 1), dtype=np.float32),
        weights=-np.ones((2, 1), dtype=np.float32),
        active=np.ones((2, 1), dtype=bool),
        dimension=np.array(1),
    )
    with pytest.raises(ValueError):
        tc.spectral.log_partition_function_from_slq(negative_measure, 0.2)
    assert np.isfinite(
        np.asarray(
            tc.spectral.log_partition_function(hamiltonian, 0.2, probes=probes[:1])
        )
    )
    with pytest.raises(ValueError, match="at least two probes"):
        tc.spectral.log_partition_function(
            hamiltonian, 0.2, probes=probes[:1], with_std=True
        )
    with pytest.raises(ValueError):
        tc.spectral.log_partition_function(
            hamiltonian, 0.2, probes=probes.reshape(1, 3, 3)
        )
    with pytest.raises(ValueError):
        tc.spectral.fft_spectrum(np.ones((2, 1)), np.array([0.0, 0.1]))
