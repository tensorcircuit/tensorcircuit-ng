import re
with open("examples/one_site_dmrg.py", "r") as f:
    code = f.read()

# fix unused variables
code = code.replace("E0_l2r, M_l2r, L_envs, final_R_mat = sweep_left_to_right", "_, M_l2r, L_envs, final_R_mat = sweep_left_to_right")
code = code.replace("final_M, final_R_envs, final_L_envs, final_E, _ = jax.lax.fori_loop", "final_M, _, _, final_E, _ = jax.lax.fori_loop")

# line too long (366): print("Final Energy Heisenberg (using quimb2qop MPO):", tc.backend.numpy(E_heis_quimb)) -> handled by black? Wait, black didn't wrap it or we have a comment?
# Let's run black first
with open("examples/one_site_dmrg.py", "w") as f:
    f.write(code)
