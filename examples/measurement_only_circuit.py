"""
Quantum Trajectory Simulation for Measurement-Induced Phase Transitions (MIPT) in Measurement-Only Models
------------------------------------------------------------------------------
This script simulates a quantum circuit under monitoring (weak measurements).
It calculates the evolution of a quantum state under:
1. Weak X measurements (single-site)
2. Weak ZZ measurements (two-site link)
3. Weak ZXZ measurements (three-site)

It utilizes TensorCircuit with a JAX backend for Just-In-Time (JIT) compilation 
to speed up the simulation of many trajectories. It computes Topological 
Entanglement Entropy (TEE) and Mutual Information (MEE).
"""

import numpy as np
import tensorcircuit as tc
from functools import partial

# Setup
K = tc.set_backend("jax")
tc.set_dtype("complex128")

# Operators
Z, X = np.array([[1,0],[0,-1]]), np.array([[0,1],[1,0]])
I8 = np.eye(8)
ZXZ = np.kron(np.kron(Z, X), Z)

@partial(K.jit, static_argnums=(3,4))
def mipt_circuit(sx, szz, szxz, n, d, cx_params, sx_params, czz_params, szz_params, czxz_params, szxz_params):
    """
    Simulates a single quantum trajectory.
    
    Args:
        status_x, status_zz, status_zxz: Random numbers used to determine measurement outcomes.
        n: Number of qubits.
        d: Circuit depth (time steps).
        cx, sx: Cosine/Sine parameters for weak X measurement strength.
        czz, szz: Cosine/Sine parameters for weak ZZ measurement strength.
        czxz, szxz: Cosine/Sine parameters for weak ZXZ measurement strength.
        
    Returns:
        x_history, zz_history, zxz_history: Record of measurement outcomes (0 or 1).
        state: The final normalized quantum state vector.
    """
    sx = K.reshape(sx, [d, n])
    szz = K.reshape(szz, [d, n])
    szxz = K.reshape(szxz, [d, n])
    
    # Initialize |+++...⟩ state
    c = tc.Circuit(n)
    for j in range(n):
        c.rx(j, theta=-np.pi/2)
    state = c.state()
    
    # A. Weak X Measurement Operators (Single Qubit)
    M1_x = 0.5*cx_params*K.convert_to_tensor(np.array([[1, 1], [1, 1]])) + \
           0.5*sx_params*K.convert_to_tensor(np.array([[1, -1], [-1, 1]]))
    M2_x = 0.5*sx_params*K.convert_to_tensor(np.array([[1, 1], [1, 1]])) + \
           0.5*cx_params*K.convert_to_tensor(np.array([[1, -1], [-1, 1]]))
    
    # B. Weak ZZ Measurement Operators (Two Qubits)
    M1_zz = czz_params*K.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 1]])) + \
            szz_params*K.convert_to_tensor(np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 0]]))
    M2_zz = szz_params*K.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 1]])) + \
            czz_params*K.convert_to_tensor(np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 0]]))
    M1_zz = M1_zz.reshape([2, 2, 2, 2])  
    M2_zz = M2_zz.reshape([2, 2, 2, 2])

    # Convert global constants to tensors
    I8_t = K.convert_to_tensor(I8)
    ZXZ_t = K.convert_to_tensor(ZXZ)

    # C. Weak ZXZ Measurement Operators (Three Qubits)
    M1_zxz = 0.5 * (czxz_params * (I8_t + ZXZ_t) + szxz_params * (I8_t - ZXZ_t))
    M2_zxz = 0.5 * (szxz_params * (I8_t + ZXZ_t) + czxz_params * (I8_t - ZXZ_t))
    M1_zxz = M1_zxz.reshape([2, 2, 2, 2, 2, 2])  
    M2_zxz = M2_zxz.reshape([2, 2, 2, 2, 2, 2])
    
    x_h, zz_h, zxz_h = [], [], []
    
    for t in range(d):
        c = tc.Circuit(n, inputs=state)
        
        # ZZ measurements (alternating bonds)
        for i in range(t%2, n-1, 2):
            zz_h.append(c.general_kraus([M1_zz, M2_zz], i, i+1, status=szz[t,i], with_prob=False))
        state = c.state()/K.norm(c.state())
        c = tc.Circuit(n, inputs=state)
        
        # ZXZ measurements (3-qubit blocks)
        for i in range(t%3, n-2, 3):
            zxz_h.append(c.general_kraus([M1_zxz, M2_zxz], i, i+1, i+2, status=szxz[t,i], with_prob=False))
        state = c.state()/K.norm(c.state())
        c = tc.Circuit(n, inputs=state)
        
        # X measurements (all qubits)
        for i in range(n):
            x_h.append(c.general_kraus([M1_x, M2_x], i, status=sx[t,i], with_prob=False))
        state = c.state()/K.norm(c.state())
    
    return K.stack(x_h), K.stack(zz_h), K.stack(zxz_h), state

# --- Entanglement Metrics ---

def TEE(state, n):
    """Topological Entanglement Entropy: S_AB + S_BC - S_B - S_D"""
    p = n//4
    A,B,D,C = range(0,p), range(p,2*p), range(2*p,3*p), range(3*p,n)
    
    rho_AB = tc.quantum.reduced_density_matrix(state, cut=list(C)+list(D))
    rho_BC = tc.quantum.reduced_density_matrix(state, cut=list(A)+list(D))
    rho_B = tc.quantum.reduced_density_matrix(state, cut=list(A)+list(C)+list(D))
    rho_D = tc.quantum.reduced_density_matrix(state, cut=list(A)+list(B)+list(C))
    
    return (tc.quantum.entropy(rho_AB) + tc.quantum.entropy(rho_BC) 
            - tc.quantum.entropy(rho_B) - tc.quantum.entropy(rho_D))

def MEE(state, n):
    """Mutual Information between first and last qubit: S_A + S_B - S_AB"""
    rho_A = tc.quantum.reduced_density_matrix(state, cut=list(range(1,n)))
    rho_B = tc.quantum.reduced_density_matrix(state, cut=list(range(0,n-1)))
    rho_AB = tc.quantum.reduced_density_matrix(state, cut=list(range(1,n-1)))
    return tc.quantum.entropy(rho_A) + tc.quantum.entropy(rho_B) - tc.quantum.entropy(rho_AB)

# Execution
if __name__ == "__main__":
    n, d, trajectories = 6, 12, 10

    # Measurement strengths (γ)
    # γ_{a} = 0: No measurement 
    # γ_{a} = 1: Strong projective measurement with operator a
    γ_x, γ_zz = 0.01, 0.01
    
    cx, sx = np.cos((1-γ_x)*np.pi/4), np.sin((1-γ_x)*np.pi/4)
    czz, szz = np.cos((1-γ_zz)*np.pi/4), np.sin((1-γ_zz)*np.pi/4)
    czxz, szxz = np.cos((γ_zz+γ_x)*np.pi/4), np.sin((γ_zz+γ_x)*np.pi/4)
    
    I_ab, T_ee = 0, 0
    results_x, results_zz, results_zxz = [], [], []
    
    for _ in range(trajectories):
        sx_r = np.random.uniform(size=[d*n])
        szz_r = np.random.uniform(size=[d*n])
        szxz_r = np.random.uniform(size=[d*n])
        
        x, zz, zxz, state = mipt_circuit(sx_r, szz_r, szxz_r, n, d, cx, sx, czz, szz, czxz, szxz)
        
        results_x.append(x)
        results_zz.append(zz)
        results_zxz.append(zxz)
        
        I_ab += MEE(state, n)/trajectories
        T_ee += TEE(state, n)/trajectories
    
    # Output
    print("X outcomes:", np.concatenate(results_x).tolist())
    print("ZZ outcomes:", np.concatenate(results_zz).tolist())
    print("ZXZ outcomes:", np.concatenate(results_zxz).tolist())
    print(f"MI (bits): {I_ab/np.log(2):.4f}")
    print(f"TEE (bits): {T_ee/np.log(2):.4f}")
   
    
    








    



    
