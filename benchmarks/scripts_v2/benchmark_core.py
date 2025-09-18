import numpy as np
import tensorcircuit as tc
from tensorcircuit import noisemodel, channels


def generate_1d_circuit(c, params, nqubits, nlayers):
    for i in range(nqubits):
        c.h(i)
    for j in range(nlayers):
        for i in range(nqubits):
            c.rx(i, theta=params[j, i])
        for i in range(nqubits - 1):
            c.cx(i, i + 1)
    return c


def generate_2d_circuit(c, lx, ly, params, nqubits, nlayers):
    assert nqubits == lx * ly
    for i in range(nqubits):
        c.h(i)
    for j in range(nlayers):
        for i in range(nqubits):
            c.rx(i, theta=params[j, i])
        # Apply CX gates in a 2D square lattice pattern
        for x in range(lx):
            for y in range(ly):
                idx = x * ly + y
                # Horizontal CX gates (with right neighbor)
                if x < lx - 1:
                    c.cx(idx, idx + ly)
                # Vertical CX gates (with lower neighbor)
                if y < ly - 1:
                    c.cx(idx, idx + 1)
    return c


def generate_noisy_circuit(c, status, type="depolarizing"):
    noise_conf = noisemodel.NoiseConf()
    if type == "depolarizing":
        error1 = channels.depolarizingchannel(0.1, 0.1, 0.1)  # px, py, pz probabilities
    elif type == "amplitudedamping":
        error1 = channels.amplitudedampingchannel(0.2, 0.1)
    else:
        raise ValueError("Noise type not supported")
    noise_conf.add_noise("h", error1)  # After H gates
    noise_conf.add_noise("rx", error1)  # After RX gates
    # Add for other single-qubit gates as needed

    noisy_circuit = noisemodel.circuit_with_noise(c, noise_conf, status)
    return noisy_circuit


def get_sample(c):
    return c.sample(allow_state=False, batch=1)[0]


def get_state(c):
    return c.state()


def get_exps(c):
    return tc.backend.real(c.expectation_ps(z=[0], reuse=False))


# Mega benchmark function
def benchmark_mega_function(
    nqubits,
    nlayers,
    lx,
    ly,
    circuit_type="circuit",  # "circuit", "dmcircuit", "mpscircuit"
    bond_dim=16,
    layout_type="1d",  # "1d", "2d"
    operation="state",  # "state", "sample", "exps"
    noisy=False,  # True, False
    noisy_type="depolarizing",  # "depolarizing", "amplitudesdamping"
    use_grad=False,  # True, False
    use_vmap=False,  # True, False
    contractor=None,  # contractor setting like "cotengra-16-128"
    jit_compile=True,  # True, False
):
    """
    Mega benchmark function that can control all parameters via arguments.

    Args:
        nqubits: Number of qubits
        nlayers: Number of layers
        lx: Lattice size x (for 2D)
        ly: Lattice size y (for 2D)
        circuit_type: Type of circuit ("circuit", "dmcircuit", "mpscircuit")
        bond_dim: Bond dimension for MPS circuits
        layout_type: Circuit layout ("1d", "2d")
        operation: Operation to perform ("state", "sample", "exps")
        noisy: Whether to add noise (only for "circuit" and "dmcircuit")
        noisy_type: Type of noise channel ("depolarizing", "amplitudedamping")
        use_grad: Whether to compute gradient (AD)
        use_vmap: Whether to use vectorized operations
        contractor: Contractor setting like "cotengra-16-128"

    Returns:
        A function that takes parameters as input and returns the result
    """

    def circuit_func(params):
        # Create circuit based on type
        if circuit_type == "circuit":
            c = tc.Circuit(nqubits)
        elif circuit_type == "dmcircuit":
            c = tc.DMCircuit(nqubits)
        elif circuit_type == "mpscircuit":
            c = tc.MPSCircuit(nqubits)
            c.set_split_rules({"max_singular_values": bond_dim})

        # Generate circuit based on layout
        if layout_type == "1d":
            c = generate_1d_circuit(c, params, nqubits, nlayers)
        else:  # 2d
            c = generate_2d_circuit(c, lx, ly, params, nqubits, nlayers)

        # Add noise if requested and applicable
        if noisy and circuit_type in ["circuit", "dmcircuit"]:
            status = tc.backend.convert_to_tensor(np.random.uniform(size=2048))
            c = generate_noisy_circuit(c, status, noisy_type)

        # Perform operation
        if operation == "state":
            return get_state(c)
        elif operation == "sample":
            return get_sample(c)
        elif operation == "exps":
            return get_exps(c)

    # Apply contractor if specified and applicable
    if contractor is not None and circuit_type in ["circuit", "dmcircuit"]:
        circuit_func = tc.set_function_contractor(contractor)(circuit_func)

    # Handle gradient computation
    if use_grad and not use_vmap:
        grad_func = tc.backend.grad(circuit_func)
        return tc.backend.jit(grad_func, jit_compile=jit_compile)

    # Handle vmap computation
    if use_vmap and not use_grad:
        return tc.backend.jit(tc.backend.vmap(circuit_func), jit_compile=jit_compile)

    # Handle both grad and vmap
    if use_grad and use_vmap:
        vvag_func = tc.backend.vvag(circuit_func)
        return tc.backend.jit(vvag_func, jit_compile=jit_compile)

    # Regular operation (no grad, no vmap)
    # Always JIT the returned function
    return tc.backend.jit(circuit_func, jit_compile=jit_compile)
