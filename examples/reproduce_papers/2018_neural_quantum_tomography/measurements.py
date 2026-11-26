"""Bond-two W-state sampling in Z and adjacent XX/XY bases."""

import numpy as np
import tensorcircuit as tc

SAMPLE_BLOCK = 1024


def pack_bits(bits):
    """Pack site zero into the most significant bit of a uint32."""
    shifts = np.arange(bits.shape[-1] - 1, -1, -1, dtype=np.uint32)
    return np.sum(np.asarray(bits, dtype=np.uint32) << shifts, axis=-1, dtype=np.uint32)


def unpack_bits(indices, n):
    """Restore site-ordered measurement bits."""
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint32)
    return (np.asarray(indices, dtype=np.uint32)[..., None] >> shifts) & 1


def w_tensors(phases):
    """Right-canonical MPS with a bond index recording whether a 1 was emitted."""
    n = len(phases)
    tensors = []
    for site, angle in enumerate(phases):
        remaining = n - site
        tensor = np.zeros((2, 2, 2), dtype=tc.dtypestr)
        tensor[0, 0, 0] = np.sqrt((remaining - 1) / remaining)
        tensor[0, 1, 1] = np.exp(1j * angle) / np.sqrt(remaining)
        tensor[1, 0, 1] = 1
        tensors.append(tensor)
    tensors[0] = tensors[0][:1, :, :]
    tensors[-1] = tensors[-1][:, :, 1:]
    return tuple(tc.backend.convert_to_tensor(tensor) for tensor in tensors)


def basis_tensors(tensors, site, use_y):
    """Rotate a neighboring pair, leaving the MPS canonical at site zero."""
    circuit = tc.MPSCircuit(len(tensors), tensors=tensors, center_position=0)
    circuit.h(site)
    if use_y:
        circuit.sd(site + 1)
    circuit.h(site + 1)
    circuit.position(0)
    return tuple(circuit.get_tensors())


def make_sampler():
    """Compile single-shot sampling with shared, dynamically supplied tensors."""
    K = tc.backend

    def sample_one(tensors, status):
        circuit = tc.MPSCircuit(len(tensors), tensors=tensors, center_position=0)
        return circuit.sample(status=status)

    return K.jit(K.vmap(sample_one, vectorized_argnums=1))


def sample_basis(sampler, tensors, shots, rng):
    """Draw one basis in fixed-size blocks to reuse the compiled shape."""
    K = tc.backend
    result = np.empty(shots, dtype=np.uint32)
    for start in range(0, shots, SAMPLE_BLOCK):
        take = min(shots - start, SAMPLE_BLOCK)
        status = rng.random((SAMPLE_BLOCK, len(tensors)), dtype=np.float32)
        bits = K.numpy(sampler(tensors, K.convert_to_tensor(status)))[:take]
        result[start : start + take] = pack_bits(bits)
    return result


def sample_bases(sampler, phases, shots, rng):
    """Return Z followed by XX and XY for each edge in order."""
    tensors = w_tensors(phases)
    rows = [sample_basis(sampler, tensors, shots, rng)]
    for site in range(len(phases) - 1):
        for use_y in (False, True):
            rotated = basis_tensors(tensors, site, use_y)
            rows.append(sample_basis(sampler, rotated, shots, rng))
    return np.stack(rows)
