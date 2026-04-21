"""
Surface Code TN Decoder using Stim DEM
This script demonstrates a general parser that translates a Stim Detector Error Model (DEM)
into a Tensor Network for maximum likelihood decoding. It uses JAX JIT and lax.map
(sequential loop) for high-performance and memory-efficient decoding.
"""

import itertools
from functools import partial
from typing import Tuple
import stim
import numpy as np
import tensornetwork as tn
import jax
import jax.numpy as jnp
import tensorcircuit as tc

# Set global precision to complex128; tc.rdtypestr will then be float64
tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra-16-32")


def parse_dem(dem: stim.DetectorErrorModel):
    """
    Parse stim DEM into flat structures suitable for JIT.
    """
    error_ps = []
    error_dets = []
    error_obs = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            dets = []
            obs = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    dets.append(target.val)
                elif target.is_logical_observable_id():
                    obs.append(target.val)
            if 0 < p < 1:
                error_ps.append(p)
                error_dets.append(tuple(dets))
                error_obs.append(tuple(obs))

    return (
        jnp.array(error_ps, dtype=tc.rdtypestr),
        tuple(error_dets),
        tuple(error_obs),
        dem.num_detectors,
        dem.num_observables,
    )


def compute_likelihood(
    error_ps: jnp.ndarray,
    error_dets: Tuple[Tuple[int, ...], ...],
    error_obs: Tuple[Tuple[int, ...], ...],
    num_detectors: int,
    num_observables: int,
    detector_syndrome: jnp.ndarray,
    observable_twists: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build and contract the TN for a specific logical sector.
    All tensors are initialized in real precision (tc.rdtypestr).
    """
    nodes = []
    # Normalize Hadamard to preserve tensor norm and prevent overflow/underflow
    hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=tc.rdtypestr) / jnp.sqrt(2.0)

    # 1. Detector Nodes (Hyper-edges)
    det_to_errors = [[] for _ in range(num_detectors)]
    for j, dets in enumerate(error_dets):
        for d_idx in dets:
            det_to_errors[d_idx].append(j)

    det_nodes = []
    for i in range(num_detectors):
        m = detector_syndrome[i]
        factor = jnp.array([1.0, (-1.0) ** m], dtype=tc.rdtypestr)
        degree = len(det_to_errors[i]) + 1
        sn = tn.CopyNode(degree, 2, name=f"D{i}")
        syn_node = tn.Node(factor, name=f"SynD{i}")
        nodes.append(syn_node)
        syn_node[0] ^ sn[0]
        det_nodes.append(sn)

    # 2. Observable Nodes (Hyper-edges)
    obs_to_errors = [[] for _ in range(num_observables)]
    for j, obs in enumerate(error_obs):
        for o_idx in obs:
            obs_to_errors[o_idx].append(j)

    obs_nodes = []
    for k in range(num_observables):
        l = observable_twists[k]
        factor = jnp.array([1.0, (-1.0) ** l], dtype=tc.rdtypestr)
        degree = len(obs_to_errors[k]) + 1
        on = tn.CopyNode(degree, 2, name=f"L{k}")
        twist_node = tn.Node(factor, name=f"TwistL{k}")
        nodes.append(twist_node)
        twist_node[0] ^ on[0]
        obs_nodes.append(on)

    # 3. Error Nodes
    det_leg_counters = [1] * num_detectors
    obs_leg_counters = [1] * num_observables
    err_nodes = []

    for j, p in enumerate(error_ps):
        prob_vec = jnp.array([1.0 - p, p], dtype=tc.rdtypestr)
        dets, obs = error_dets[j], error_obs[j]
        degree = len(dets) + len(obs) + 1
        en = tn.CopyNode(degree, 2, name=f"E{j}")
        err_nodes.append(en)
        prob_node = tn.Node(prob_vec, name=f"ProbE{j}")
        nodes.append(prob_node)
        prob_node[0] ^ en[0]

        curr_leg = 1
        for d_idx in dets:
            h_node = tn.Node(hadamard, name=f"H_E{j}_D{d_idx}")
            nodes.append(h_node)
            en[curr_leg] ^ h_node[0]
            h_node[1] ^ det_nodes[d_idx][det_leg_counters[d_idx]]
            det_leg_counters[d_idx] += 1
            curr_leg += 1

        for o_idx in obs:
            h_node = tn.Node(hadamard, name=f"H_E{j}_L{o_idx}")
            nodes.append(h_node)
            en[curr_leg] ^ h_node[0]
            h_node[1] ^ obs_nodes[o_idx][obs_leg_counters[o_idx]]
            obs_leg_counters[o_idx] += 1
            curr_leg += 1

    # Final contraction
    all_nodes = nodes + det_nodes + obs_nodes + err_nodes
    result = tc.contractor(all_nodes)
    return result.tensor


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def decode_jit(
    error_ps: jnp.ndarray,
    error_dets: Tuple[Tuple[int, ...], ...],
    error_obs: Tuple[Tuple[int, ...], ...],
    num_detectors: int,
    num_observables: int,
    detector_syndrome: jnp.ndarray,
) -> jnp.ndarray:
    """
    JITed decoding function using lax.map (sequential loop) for memory efficiency.
    """
    sectors = jnp.array(list(itertools.product([0, 1], repeat=num_observables)))

    def single_sector_likelihood(s):
        return compute_likelihood(
            error_ps,
            error_dets,
            error_obs,
            num_detectors,
            num_observables,
            detector_syndrome,
            s,
        )

    # lax.map ensures O(1) memory complexity relative to the number of sectors.
    likelihoods = jax.lax.map(single_sector_likelihood, sectors)

    # Guard against division by zero
    total = jnp.sum(likelihoods)
    return jnp.where(total > 0, likelihoods / total, likelihoods)


def main():
    d, rounds, p = 3, 3, 0.01
    print(
        f"--- Surface Code DEM TN Decoder (JIT+lax.map, d={d}, rounds={rounds}, p={p}) ---"
    )

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    dem = circuit.detector_error_model(flatten_loops=True)
    error_ps, error_dets, error_obs, n_det, n_obs = parse_dem(dem)

    sampler = circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(1, separate_observables=True)
    detector_syndrome = jnp.array(det_samples[0].astype(int))
    actual_obs_flip = obs_samples[0].astype(int)

    print(f"DEM: {n_det} detectors, {n_obs} observables, {len(error_ps)} error events")
    print(f"Actual logical flips: {actual_obs_flip}")

    print("\nDecoding (Sequential sectors via lax.map)...")
    probs = decode_jit(error_ps, error_dets, error_obs, n_det, n_obs, detector_syndrome)

    sectors = list(itertools.product([0, 1], repeat=n_obs))
    for i, sector in enumerate(sectors):
        print(f"  Sector {sector}: {probs[i]:.6e}")

    best_sector = sectors[np.argmax(probs)]
    print(f"\nDecoded logical flips: {best_sector}")
    if np.array_equal(best_sector, actual_obs_flip):
        print("[SUCCESS] Decoder correctly identified the logical flip.")


if __name__ == "__main__":
    main()
