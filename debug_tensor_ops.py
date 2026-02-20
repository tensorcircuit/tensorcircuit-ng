
import jax.numpy as jnp
import numpy as np

def main():
    # Simulate update_R for i=3 (Boundary)
    rows = 4
    K = 2

    # R initialized to 1s.
    # Shape: (b_old, b_new) + (g0_in, g0_out) + (g1_in, g1_out)
    shape_R = (1, 1, 1, 1, 1, 1)
    T = jnp.ones(shape_R)

    # update_R logic simulation
    # Contract with A (dummy) -> adds phys
    # T becomes (b, b, p_new, p_old, g...)
    # p dims are 2.
    shape_T = (1, 1) + (2,)*4 + (2,)*4 + (1, 1, 1, 1)
    T = jnp.ones(shape_T)

    print(f"T init shape: {T.shape}")

    g_start = 10 # 2 + 8

    # Loop k=1. gate_L exists.
    k = 1
    idx_ri = g_start + 2*k # 12
    idx_ro = g_start + 2*k + 1 # 13

    r = 0 # gate_L index 1 % rows
    phys_old_start = 6

    # Gate L logic
    # T = tensordot(T, U, axes=[[phys_old+r], [3]])
    # U is (2, 2, 2, 2). axes=[[6], [3]].
    # T rank 14. U rank 4. Result 16.
    # Removes T[6], U[3].
    # T indices: 0..5, 7..13, U[0], U[1], U[2].
    # idx_ri (12) becomes 11. idx_ro (13) becomes 12.
    # U[0, 1, 2] are appended (indices 13, 14, 15).
    # New phys is U[2] (15).
    # New legs are U[0, 1] (13, 14).

    # Simulating shape
    new_shape = list(T.shape)
    new_shape.pop(6) # phys
    new_shape.extend([2, 2, 2]) # lo, ro, li
    # Note: U indices lo, ro, li.
    # logic: lo, ro, li.

    print(f"After dot shape: {new_shape}")

    # moveaxis(T, -2, phys_old_start+r)
    # Move -2 (ro?) to 6.
    # Wait, my code logic:
    # T = tensordot(T, U, axes=[[phys_old_start+r], [3]])
    # T: (..., lo, ro, li).
    # moveaxis -2 (ro) to phys.
    # ro is new phys?
    # U(lo, ro, li, ri).
    # Contract ri (3).
    # Remaining: lo(0), ro(1), li(2).
    # ro(1) is new phys?
    # Logic in code: "ro is new phys_old".
    # So yes.
    # move -2 (ro) to 6.

    # New legs: lo(0), li(2).
    # lo is at -3. li is at -1.
    # lo matches idx_ro. li matches idx_ri.
    # squeeze idx_ri, idx_ro.
    # move -1 to idx_ri. move -2 to idx_ro.

    # Let's perform using JAX
    T = jnp.ones(shape_T)
    U = jnp.ones((2, 2, 2, 2))

    T = jnp.tensordot(T, U, axes=[[6], [3]])
    # T shape: ..., lo, ro, li
    print(f"After dot: {T.shape}")

    T = jnp.moveaxis(T, -2, 6)
    # T shape: ..., lo, li. (ro moved to 6).
    print(f"After move phys: {T.shape}")

    # squeeze 12, 13
    T = jnp.squeeze(T, axis=(12, 13))
    print(f"After squeeze: {T.shape}")

    # move -1 (li) to 12. -2 (lo) to 13.
    T = jnp.moveaxis(T, [-1, -2], [12, 13])
    print(f"After move legs: {T.shape}")

    # Check dims at 12, 13
    print(f"Dims at 12, 13: {T.shape[12]}, {T.shape[13]}")

    # k=0. pass.

    # Final trace
    for r in range(4):
        # trace 2, 6
        T = jnp.trace(T, axis1=2, axis2=6)
        print(f"Trace {r}: {T.shape}")

    print(f"Final shape: {T.shape}")

if __name__ == "__main__":
    main()
