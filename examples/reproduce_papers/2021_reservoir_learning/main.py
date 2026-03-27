"""Reproduction of "The Reservoir Learning Power across Quantum Many-Boby Localization Transition"
Link: https://arxiv.org/abs/2104.02727
Description:
This script reproduces Figure 5(b) from the paper using TensorCircuit-NG. The simulation
is scaled down to N=8 qubits, with shorter sequence lengths and fewer samples for feasibility,
while preserving the core MBL transition phenomenon in the critical prediction length l_c.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import jax
import tensorcircuit as tc

jax.config.update("jax_enable_x64", True)
tc.set_backend("jax")

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
# Scaled down parameters
N = 6  # Number of qubits (reduced further to speed up execution)
K1 = 100  # Discard steps
K2 = 300  # Training steps
K3_max = 15  # Max prediction steps
samples = 2  # Number of disorder samples
W_range = np.linspace(1, 15, 6)  # Disorder strengths to sweep
alphas = [0.4, 0.8, 1.2]

# Fixed physical parameters
B_field = 4.0
J0 = 1.0
J0_tau = 2.0
V = 10  # subintervals
tau = J0_tau / J0
delta_t = tau / V
noise_sigma = 1e-6

# MG Parameters
gamma = 0.9
beta = 10.0
lam = 0.2
k_delta = 17


def generate_mackey_glass(total_length):
    """
    Generates a scaled Mackey-Glass time sequence.
    """
    F = np.zeros(total_length + k_delta + 1)

    # Initialize with small random values to start chaos
    F[: k_delta + 1] = np.random.uniform(0.5, 1.5, size=k_delta + 1)

    for k in range(k_delta, total_length + k_delta):
        F[k + 1] = gamma * F[k] + (lam * F[k - k_delta]) / (1 + F[k - k_delta] ** beta)

    # Discard initial transient and get the target length
    F_seq = F[k_delta + 1 :]

    # Rescale to [0, 1] as per the paper
    F_min = np.min(F_seq)
    F_max = np.max(F_seq)
    s_k = (F_seq - F_min) / (F_max - F_min)

    return s_k


def build_hamiltonian(alpha, W, phi):
    """
    Builds the disordered long-range Ising Hamiltonian matrix
    H = sum_{i<j} J0 |i-j|^-alpha X_i X_j + 1/2 sum_i (B + phi_i) Z_i
    """
    H = np.zeros((2**N, 2**N), dtype=np.complex128)

    # X_i X_j terms
    for i in range(N):
        for j in range(i + 1, N):
            J_ij = J0 * np.abs(i - j) ** (-alpha)

            # Construct X_i X_j matrix
            term = 1
            for k in range(N):
                if k == i or k == j:
                    term = np.kron(term, tc.gates._x_matrix)
                else:
                    term = np.kron(term, tc.gates._i_matrix)

            H += J_ij * term

    # Z_i terms
    for i in range(N):
        term = 1
        for k in range(N):
            if k == i:
                term = np.kron(term, tc.gates._z_matrix)
            else:
                term = np.kron(term, tc.gates._i_matrix)

        H += 0.5 * (B_field + phi[i]) * term

    return tc.backend.convert_to_tensor(H)


def get_unitary(H, dt):
    """
    Returns the time evolution unitary U = exp(-i H dt)
    """
    dt_tensor = tc.backend.convert_to_tensor(dt, dtype="complex128")
    return tc.backend.expm(-1j * tc.backend.cast(H, dtype="complex128") * dt_tensor)


def inject_input(rho, s_k):
    """
    Injects the input signal s_k into the first qubit.
    rho -> |psi_sk><psi_sk| tensor Tr_1[rho]
    """
    # Tr_1[rho]
    # Tracing out the first qubit: rho is shape (2**N, 2**N)
    # Reshape to (2, 2**(N-1), 2, 2**(N-1))
    rho_reshaped = tc.backend.reshape(rho, (2, 2 ** (N - 1), 2, 2 ** (N - 1)))
    # Trace out the first subsystem
    rho_tr1 = tc.backend.einsum("ijik->jk", rho_reshaped)

    # |psi_sk> = sqrt(1-sk)|+> + sqrt(sk)|->
    # |+> = [1/sqrt(2), 1/sqrt(2)]
    # |-> = [1/sqrt(2), -1/sqrt(2)]
    # We will compute the density matrix directly
    s_k_val = tc.backend.numpy(s_k)  # safe fallback
    p_plus = np.sqrt(1 - s_k_val)
    p_minus = np.sqrt(s_k_val)
    psi_sk = p_plus * np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]) + p_minus * np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)]
    )
    rho_sk = np.outer(psi_sk, psi_sk.conj())
    rho_sk_tensor = tc.backend.convert_to_tensor(rho_sk, dtype="complex128")

    # New rho
    rho_new = tc.backend.kron(rho_sk_tensor, rho_tr1)
    return rho_new


def measure_X(rho):
    """
    Measures <X_i> for all qubits.
    """
    X_matrix = tc.backend.convert_to_tensor(tc.gates._x_matrix, dtype="complex128")
    I_matrix = tc.backend.convert_to_tensor(tc.gates._i_matrix, dtype="complex128")

    measurements = []
    for i in range(N):
        obs = tc.backend.convert_to_tensor(1.0, dtype="complex128")
        for k in range(N):
            if k == i:
                obs = tc.backend.kron(obs, X_matrix)
            else:
                obs = tc.backend.kron(obs, I_matrix)

        val = tc.backend.real(tc.backend.trace(tc.backend.matmul(rho, obs)))
        measurements.append(val)

    return tc.backend.stack(measurements)


def run_reservoir(alpha, W, s_seq):
    """
    Runs the quantum reservoir for a single sample.
    """
    # 1. Initialize random field for this sample
    phi = np.random.uniform(-W / 2, W / 2, size=N)

    # 2. Build Hamiltonian and Unitary for subintervals
    H = build_hamiltonian(alpha, W, phi)
    U = get_unitary(H, delta_t)
    U_dag = tc.backend.conj(tc.backend.transpose(U))

    # 3. Initialize rho = I / 2^N (infinite temperature state)
    rho = tc.backend.convert_to_tensor(np.eye(2**N) / (2**N), dtype="complex128")

    # We will store signals during K2 (training) and K3 (prediction context)
    S_train = []  # Shape will be (K2, V*N)
    y_star_train = []  # Target: s_{k+1}

    # Evolve through K1 (discard)
    for k in range(K1):
        s_k = tc.backend.convert_to_tensor(s_seq[k], dtype="complex128")
        rho = inject_input(rho, s_k)
        for _ in range(V):
            rho = U @ rho @ U_dag
            # No need to store measurements during K1

    # Evolve through K2 (training)
    for k in range(K1, K1 + K2):
        s_k = tc.backend.convert_to_tensor(s_seq[k], dtype="complex128")
        rho = inject_input(rho, s_k)

        S_k_v = []
        for _ in range(V):
            rho = U @ rho @ U_dag
            # Measure at end of subinterval
            meas = measure_X(rho)
            S_k_v.append(tc.backend.numpy(meas))

        S_k_flat = np.concatenate(S_k_v)
        # S_{k,v,i} = (1 + <X_i>) / 2
        S_train.append((1.0 + S_k_flat) / 2.0)
        y_star_train.append(s_seq[k + 1])

    S_train = np.array(S_train)
    y_star_train = np.array(y_star_train)

    # Train Ridge Regression Model
    clf = Ridge(alpha=1e-8)
    clf.fit(S_train, y_star_train)

    # Evolve through K3 (prediction)
    # We want to find l_c, so we predict K3_max steps forward
    # The actual s_k used as input comes from the PREDICTION of the previous step
    # with a slight noise added as per the paper.
    predictions = []

    # Initial input to prediction phase is the last step of K2
    s_k_pred = s_seq[K1 + K2]

    for k in range(K1 + K2, K1 + K2 + K3_max):
        # Inject PREDICTED input (or initial truth for the very first step)
        s_k_tensor = tc.backend.convert_to_tensor(s_k_pred, dtype="complex128")
        rho = inject_input(rho, s_k_tensor)

        S_k_v = []
        for _ in range(V):
            rho = U @ rho @ U_dag
            meas = measure_X(rho)
            S_k_v.append(tc.backend.numpy(meas))

        S_k_flat = np.concatenate(S_k_v)
        S_pred = (1.0 + S_k_flat) / 2.0

        # Predict the next step
        y_k = clf.predict([S_pred])[0]
        predictions.append(y_k)

        # Add white noise to prediction for next step
        noise = np.random.uniform(-noise_sigma, noise_sigma)
        s_k_pred = np.clip(y_k + noise, 0.0, 1.0)  # ensure it remains physical

    y_star_pred = s_seq[K1 + K2 + 1 : K1 + K2 + K3_max + 1]

    return np.array(predictions), np.array(y_star_pred)


def calculate_covariance(y_pred, y_star, l):
    """
    Calculates the normalized covariance C for the first l predictions.
    """
    if l <= 1:
        return 1.0  # Trivial

    y_p = y_pred[:l]
    y_s = y_star[:l]

    std_p = np.std(y_p)
    std_s = np.std(y_s)

    if std_p == 0 or std_s == 0:
        return 0.0

    # Use corrcoef for exact Pearson correlation (avoids ddof=0 vs ddof=1 mismatch)
    C = np.corrcoef(y_s, y_p)[0, 1]
    return C


if __name__ == "__main__":
    print(f"Generating Mackey-Glass sequence (K1={K1}, K2={K2}, K3_max={K3_max})")
    # Generate enough sequence for all samples
    # We will slice differently for each sample to give it a "distinctive input signal"
    seq = generate_mackey_glass(K1 + K2 + K3_max + 100 * samples)

    output_dir = pathlib.Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        print(f"\n--- Starting simulation for alpha = {alpha} ---")

        for W in W_range:
            l_c_samples = []
            print(f"  W = {W:.2f}", end="", flush=True)

            for s in range(samples):
                # Shift sequence slightly for each sample
                start_idx = s * 50
                s_seq = seq[start_idx : start_idx + K1 + K2 + K3_max + 1]

                y_pred, y_star = run_reservoir(alpha, W, s_seq)

                # Find l_c (critical time step where C drops below 0.5)
                l_c = K3_max  # Default to max if it never drops below 0.5
                for l in range(2, K3_max + 1):
                    C = calculate_covariance(y_pred, y_star, l)
                    if C < 0.5:
                        l_c = l
                        break

                l_c_samples.append(l_c)
                print(".", end="", flush=True)

            avg_l_c = np.mean(l_c_samples)
            results[alpha].append(avg_l_c)
            print(f" | avg l_c = {avg_l_c:.2f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    for alpha in alphas:
        plt.plot(W_range, results[alpha], marker="o", label=rf"$\alpha={alpha}$")

    plt.xlabel(r"Disorder strength $W$")
    plt.ylabel(r"Critical time steps $l_c$")
    plt.title(
        f"Mackey-Glass Task Critical Time Steps\n($N={N}$, $K_1={K1}$, $K_2={K2}$, Samples={samples})"
    )
    plt.legend()
    plt.grid(True)

    out_path = output_dir / "result.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plot to {out_path}")
