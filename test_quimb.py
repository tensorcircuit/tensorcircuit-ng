import tensorcircuit as tc
import quimb.tensor as qtn

H = qtn.MPO_ham_heis(4)
print(type(H))
try:
    op = tc.quantum.quimb2qop(H)
    print("quimb2qop success")
    mpo_tensor = op.eval_matrix()
    print("MPO Matrix shape:", mpo_tensor.shape)
except Exception as e:
    print("quimb2qop failed:", e)

dmrg = qtn.DMRG2(H, bond_dims=[8])
dmrg.solve(tol=1e-6, verbosity=0)
print("Quimb energy:", dmrg.energy)
