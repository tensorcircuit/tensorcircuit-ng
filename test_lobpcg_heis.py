import jax
import jax.numpy as jnp
import tensorcircuit as tc
import quimb.tensor as qtn

tc.set_backend("jax")
tc.set_dtype("complex128")
jax.config.update('jax_enable_x64', True)

L = 4
H_quimb = qtn.MPO_ham_heis(L)
op = tc.quantum.quimb2qop(H_quimb)

W_list = []
# The output is a QuOperator. Let's inspect its nodes.
for n in op.nodes:
    print(n.tensor.shape)

# Let's extract the actual W_list from the nodes in a way to pass to DMRG.
# W_list needs to be shape (L, D, d, d, D)
# Quimb MPO tensors have shapes: left boundary (d, D, d), middle (d, D, D, d), right boundary (d, D, d)
