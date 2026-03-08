import os
import time
import json
import uuid
import itertools
import numpy as np
import jax
import optax
import tensorcircuit as tc

# Important setup for precision
jax.config.update('jax_enable_x64', True)
tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend

from objective import evaluate, ED_ENERGY, n, edges

def get_topology(topo_type):
    if topo_type == "lattice":
        return edges
    elif topo_type == "line":
        return [(i, i+1) for i in range(n-1)]
    elif topo_type == "all_to_all":
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        raise ValueError("Unknown topology")

def create_circuit(params, depth, topo_type, entangle_gate, rx_layers, ry_layers, rz_layers):
    c = tc.Circuit(n)
    topo = get_topology(topo_type)

    param_idx = 0

    for layer in range(depth):
        # Single qubit rotations
        if rx_layers:
            for i in range(n):
                c.rx(i, theta=params[param_idx])
                param_idx += 1
        if ry_layers:
            for i in range(n):
                c.ry(i, theta=params[param_idx])
                param_idx += 1
        if rz_layers:
            for i in range(n):
                c.rz(i, theta=params[param_idx])
                param_idx += 1

        # Entanglement
        for i, j in topo:
            if entangle_gate == "cnot":
                c.cnot(i, j)
            elif entangle_gate == "cz":
                c.cz(i, j)
            elif entangle_gate == "rzz":
                c.exp1(i, j, unitary=tc.gates._zz_matrix, theta=params[param_idx])
                param_idx += 1
            elif entangle_gate == "rxx":
                c.exp1(i, j, unitary=tc.gates._xx_matrix, theta=params[param_idx])
                param_idx += 1

    # Final rotations
    if rx_layers:
        for i in range(n):
            c.rx(i, theta=params[param_idx])
            param_idx += 1
    if ry_layers:
        for i in range(n):
            c.ry(i, theta=params[param_idx])
            param_idx += 1
    if rz_layers:
        for i in range(n):
            c.rz(i, theta=params[param_idx])
            param_idx += 1

    return c, param_idx

def count_params(depth, topo_type, entangle_gate, rx_layers, ry_layers, rz_layers):
    topo = get_topology(topo_type)
    num_rotations_per_layer = (int(rx_layers) + int(ry_layers) + int(rz_layers)) * n
    num_entangle_params = 0
    if entangle_gate in ["rzz", "rxx"]:
        num_entangle_params = len(topo)

    return depth * (num_rotations_per_layer + num_entangle_params) + num_rotations_per_layer

def run_experiment(config):
    np.random.seed(config["seed"])

    depth = config["depth"]
    topo_type = config["topology"]
    entangle_gate = config["entangle_gate"]
    rx_layers = config["rx_layers"]
    ry_layers = config["ry_layers"]
    rz_layers = config["rz_layers"]
    lr = config["learning_rate"]
    init_strategy = config["init_strategy"]
    max_steps = config.get("max_steps", 50) # using 50 to speed it up

    num_params = count_params(depth, topo_type, entangle_gate, rx_layers, ry_layers, rz_layers)
    if num_params == 0:
        return {"status": "failed", "reason": "No parameters to optimize."}

    if init_strategy == "normal_small":
        init_params = np.random.normal(0, 0.01, size=num_params)
    elif init_strategy == "normal_large":
        init_params = np.random.normal(0, np.pi, size=num_params)
    elif init_strategy == "zero":
        init_params = np.zeros(num_params)
    elif init_strategy == "symmetry_breaking":
        init_params = np.random.normal(0, 0.1, size=num_params)
        init_params += 0.5
    else:
        init_params = np.random.normal(0, 0.1, size=num_params)

    params = K.convert_to_tensor(init_params, dtype="float64")

    # Define the loss function
    def loss_fn(p):
        def circuit_fn(p_inner):
            c, _ = create_circuit(p_inner, depth, topo_type, entangle_gate, rx_layers, ry_layers, rz_layers)
            return c
        return evaluate(circuit_fn, p)

    # JIT compile the value and gradient
    try:
        vg_fn = K.jit(K.value_and_grad(loss_fn))
    except Exception as e:
        return {"status": "failed", "reason": f"JIT Compilation failed: {str(e)}"}

    # Setup optimizer
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, state):
        val, grad = vg_fn(p)
        updates, new_state = opt.update(grad, state, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, val

    t0 = time.time()
    best_energy = float('inf')

    # Optimization loop
    for step_idx in range(max_steps):
        params, opt_state, val = step(params, opt_state)
        val_scalar = float(val)
        if val_scalar < best_energy:
            best_energy = val_scalar

        # Optional: early stopping if grad is extremely small or if energy is essentially ED energy
        if np.abs(best_energy - ED_ENERGY) < 1e-4:
            break

    total_time = time.time() - t0

    return {
        "status": "success",
        "best_energy": best_energy,
        "energy_error": best_energy - ED_ENERGY,
        "num_params": num_params,
        "time": total_time,
        "steps_taken": step_idx + 1
    }

def main():
    ledger_path = "/app/examples/meta_exploration/20250101_vqe_heisenberg_2d/ledger.json"
    if os.path.exists(ledger_path):
        with open(ledger_path, "r") as f:
            try:
                ledger = json.load(f)
                if isinstance(ledger, dict):
                    ledger = []
            except json.JSONDecodeError:
                ledger = []
    else:
        ledger = []

    print(f"Target ED Energy: {ED_ENERGY:.4f}", flush=True)

    # To ensure we hit exactly 100 within a reasonable timeframe (some configs may timeout previously)
    # Generate 100 random configs
    np.random.seed(1234)

    configs = []

    while len(configs) < 100:
        topo = np.random.choice(["lattice", "line", "all_to_all"])
        gate = np.random.choice(["cnot", "cz", "rzz", "rxx"])
        depth = int(np.random.choice([1, 2, 4])) # max depth 4 to avoid overly long executions
        rx = bool(np.random.choice([True, False]))
        ry = bool(np.random.choice([True, False]))
        rz = bool(np.random.choice([True, False]))
        init_strat = np.random.choice(["normal_small", "normal_large", "zero", "symmetry_breaking"])
        lr = float(np.random.choice([0.01, 0.05, 0.1]))

        # Ensure at least one rotation layer
        if not (rx or ry or rz):
            ry = True

        # Check if already generated or in ledger
        c = {
            "id": str(uuid.uuid4())[:8],
            "topology": topo,
            "entangle_gate": gate,
            "depth": depth,
            "rx_layers": rx,
            "ry_layers": ry,
            "rz_layers": rz,
            "init_strategy": init_strat,
            "learning_rate": lr,
            "seed": int(np.random.randint(0, 10000)),
            "max_steps": 25 # lower steps to 25 to guarantee all 100 experiments finish
        }

        # Check if identical config exists (ignoring id/seed)
        c_hash = f"{topo}_{gate}_{depth}_{rx}_{ry}_{rz}_{init_strat}_{lr}"

        if not any(f"{l['config']['topology']}_{l['config']['entangle_gate']}_{l['config']['depth']}_{l['config']['rx_layers']}_{l['config']['ry_layers']}_{l['config']['rz_layers']}_{l['config']['init_strategy']}_{l['config']['learning_rate']}" == c_hash for l in ledger):
            configs.append(c)
            # Add dummy to ledger to prevent duplicates
            ledger.append({"config": c, "results": {"status": "pending"}})

    print(f"Running {len(configs)} new experiments...", flush=True)

    for i, config in enumerate(configs):
        print(f"Exp {i+1}/{len(configs)} | ID: {config['id']} | Topo: {config['topology']} | Gate: {config['entangle_gate']} | Depth: {config['depth']} | Init: {config['init_strategy']}", flush=True)

        try:
            res = run_experiment(config)
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            res = {"status": "failed", "reason": str(e)}

        result_record = {
            "config": config,
            "results": res
        }

        print(f"  Result: {res.get('status')} | Energy: {res.get('best_energy', 'N/A')} | Time: {res.get('time', 'N/A'):.2f}s", flush=True)

        # Update ledger correctly
        for j, l in enumerate(ledger):
            if l["config"]["id"] == config["id"]:
                ledger[j] = result_record
                break

        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

        # Save snapshot for successful runs
        if res["status"] == "success":
            snapshot_path = f"/app/examples/meta_exploration/20250101_vqe_heisenberg_2d/.snapshots/{config['id']}.json"
            with open(snapshot_path, "w") as f:
                json.dump(result_record, f, indent=2)

if __name__ == "__main__":
    main()
