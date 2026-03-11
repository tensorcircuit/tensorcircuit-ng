import jax
import jax.numpy as jnp
import tensorcircuit as tc
import quimb.tensor as qtn
import numpy as np

tc.set_backend("jax")
tc.set_dtype("complex128")
jax.config.update('jax_enable_x64', True)

def get_quimb_mpo(L, chi, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    # Generating quimb MPO
    H_quimb = qtn.MPO_ham_heis(L)
    # The arrays in H_quimb can be extracted directly
    # Shape convention in quimb for 1D MPO:
    # Left: (D, d_out, d_in)  -> Actually looking at the print: (D, d, d)
    # Middle: (D_L, D_R, d, d)
    # Right: (D, d, d)

    W_list = []
    # Let's extract them directly
    for i in range(L):
        t = H_quimb.tensors[i].data
        # We need W to be (D_L, d_out, d_in, D_R)
        # We can pad to a uniform D=5
        W = np.zeros((5, 2, 2, 5), dtype=np.complex128)

        if i == 0:
            # left boundary: t is (D, d, d) or something
            # quimb convention: first leg is right bond, then physical legs.
            # D_R = 5, d_out = 2, d_in = 2
            # wait, let's look at shape: (5, 2, 2). Usually it's (D_R, d_out, d_in)
            W[4, :, :, :] = t.transpose((1, 2, 0)) # assuming d, d, D_R or D_R, d, d
        elif i == L - 1:
            # right boundary: (D_L, d, d) -> (5, 2, 2)
            W[:, :, :, 0] = t.transpose(...)
        else:
            # middle: (D_L, D_R, d, d) or something
            W[:, :, :, :] = ...
