"""
Challenge Suite Problem 12: variational circuit to MPS overlap optimization.

The solution contracts a DMRG-MPS target bra directly with a trainable circuit
ket and differentiates the scalar overlap loss with respect to circuit angles.
"""

import os

_XLA_FAST_COMPILE_FLAG = "--xla_disable_hlo_passes=fusion"
if _XLA_FAST_COMPILE_FLAG not in os.environ.get("XLA_FLAGS", "").split():
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " " + _XLA_FAST_COMPILE_FLAG
    ).strip()

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def run_solution(config):
    n_qubits = config["n_qubits"]
    parameter_count = 0
    for layer in range(config["n_layers"]):
        parameter_count += 15 * len(range(layer % 2, n_qubits - 1, 2))

    rng = np.random.default_rng(config["seed"])
    params = rng.normal(
        scale=config["initial_parameter_scale"],
        size=(parameter_count,),
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    target_mps = tc.quantum.quimb2qop(config["dmrg_state"])
    target_bra = target_mps.adjoint()
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        circuit = tc.Circuit(n_qubits)
        for i in range(1, n_qubits, 2):
            circuit.x(i)

        offset = 0
        for layer in range(config["n_layers"]):
            for i in range(layer % 2, n_qubits - 1, 2):
                circuit.su4(i, i + 1, theta=p[offset : offset + 15])
                offset += 15

        overlap_value = (target_bra @ circuit.quvector()).eval()
        fidelity = K.real(K.conj(overlap_value) * overlap_value)
        return 1.0 - fidelity, (fidelity, overlap_value)

    value_and_grad = K.jit(K.value_and_grad(objective, has_aux=True))

    loss_history = []
    fidelity_history = []
    for _ in range(config["max_steps"]):
        (loss, aux), grads = value_and_grad(params)
        fidelity, overlap_value = aux
        loss_history.append(loss)
        fidelity_history.append(fidelity)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "loss_history": K.numpy(K.stack(loss_history)),
        "fidelity_history": K.numpy(K.stack(fidelity_history)),
        "final_parameters": K.numpy(params),
        "final_overlap_phase": np.asarray(np.angle(K.numpy(overlap_value))),
        "final_grad_norm": K.numpy(K.norm(grads)),
    }
