"""
This script demonstrates the classical shadow to ML pipeline using TensorCircuit.
It showcases how to:
1. Generate ground states for the Transverse Field Ising Model (TFIM) across two phases.
2. Simulate randomized Pauli measurements to generate classical shadows.
3. Use the shadow data to train a simple Neural Network (PyTorch) for phase classification.
"""

import time
import numpy as np
import scipy.sparse.linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import tensorcircuit as tc
from tensorcircuit import shadows
from tensorcircuit.templates.graphs import Line1D

# Use JAX backend for performance if available
tc.set_backend("jax")
print(f"Using backend: {tc.backend.name}")

# --- 1. Physics Simulation: TFIM Ground States ---


def get_tfim_hamiltonian(n, g, J=1.0):
    """
    Constructs the TFIM Hamiltonian using tc.quantum.heisenberg_hamiltonian.
    H = -J * sum(Z_i Z_{i+1}) - g * sum(X_i)
    """
    graph = Line1D(n)
    # heisenberg_hamiltonian args: hzz, hxx, hyy, hz, hx, hy
    # We want -J ZZ and -g X.
    # So hzz = -J, hxx = 0, hyy = 0, hx = -g, hz = 0, hy = 0
    # Returns a sparse matrix (default sparse=True)
    # We set numpy=True to get a format compatible with scipy
    return tc.quantum.heisenberg_hamiltonian(
        graph, hzz=-J, hxx=0, hyy=0, hx=-g, hz=0, hy=0, sparse=True, numpy=True
    )


def get_ground_state(n, g):
    """Computes the ground state vector using scipy sparse solver."""
    h = get_tfim_hamiltonian(n, g)
    # Use scipy sparse eigensolver
    # k=1 for ground state, which='SA' (Smallest Algebraic)
    e, v = scipy.sparse.linalg.eigsh(h, k=1, which="SA")
    del e  # e is unused
    # v is (dim, k), we want 1D array
    v = v[:, 0]
    # Convert to TensorCircuit tensor
    return tc.backend.convert_to_tensor(v)


# --- 2. Shadow Generation ---


def generate_shadow_data(n, g_points, n_snapshots_per_point):
    """
    Generates shadow data using raw measurement features.
    Returns:
        X: Tensor of shape (total_samples, 2 * n_qubits)
           Feature vector: [basis_0, ..., basis_{n-1}, outcome_0, ..., outcome_{n-1}]
           Basis: 1=X, 2=Y, 3=Z. Outcome: 0, 1.
        y: Tensor of shape (total_samples,)
    """
    all_snapshots = []
    labels = []

    print("Generating training data...")
    for g in g_points:
        label = 0 if g < 1.0 else 1
        phase_name = "Ordered" if label == 0 else "Disordered"

        # 1. Get State
        psi = get_ground_state(n, g)

        # 2. Random Measurement Basis
        # 1=X, 2=Y, 3=Z
        pauli_strings = np.random.randint(1, 4, size=(n_snapshots_per_point, n))
        pauli_strings_tc = tc.backend.convert_to_tensor(pauli_strings)

        # 3. Simulate Measurements
        # Returns binary outcomes (0 or 1)
        # shape: (ns, repeat, nq) -> (ns, nq)
        snapshots = shadows.shadow_snapshots(
            psi, pauli_strings_tc, measurement_only=True
        )
        if len(snapshots.shape) == 3:
            snapshots = snapshots[:, 0, :]

        snapshots_np = tc.backend.numpy(snapshots)

        # 4. Construct Features: Concatenate Basis and Outcome
        # Shape: (ns, 2*n)
        # [P1, P2, ..., Pn, O1, O2, ..., On]
        features = np.concatenate([pauli_strings, snapshots_np], axis=1)

        all_snapshots.append(features)
        labels.extend([label] * n_snapshots_per_point)

        print(f"  Processed g={g:.2f} ({phase_name})")

    X = np.vstack(all_snapshots)
    y = np.array(labels)

    # Optional: We don't necessarily need scaling for categorical inputs (Basis)
    # but outcomes are 0/1.
    # Let's keep it simple without scaling or maybe just scale.
    # Remove scaling for categorical data (Basis 1/2/3)
    # The neural network with embeddings will handle this raw data better.

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# --- 3. ML Model & Training ---


class ShadowClassifier(nn.Module):
    def __init__(self, n_qubits, hidden_dim=64):
        super().__init__()
        self.n = n_qubits
        # Embedding for Pauli Basis (0(pad), 1(X), 2(Y), 3(Z)) -> 4 indices
        # We embed each basis choice into a vector of size 8
        self.embedding = nn.Embedding(4, 4)

        # Input features: (n_qubits * 8) [embedded basis] + n_qubits [outcomes]
        input_dim = n_qubits * 4 + n_qubits

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x shape: (batch, 2*n)
        # Split into Basis (first n) and Outcome (last n)
        basis_indices = x[:, : self.n].long()  # (batch, n)
        outcomes = x[:, self.n :]  # (batch, n)

        # Embed basis: (batch, n, 4)
        embedded_basis = self.embedding(basis_indices)
        # Flatten embedding: (batch, n*4)
        embedded_basis = embedded_basis.view(embedded_basis.size(0), -1)

        # Concatenate with outcomes
        features = torch.cat([embedded_basis, outcomes], dim=1)

        return self.net(features)


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Infer n_qubits from feature shape (2 * n)
    n_qubits = X.shape[1] // 2
    model = ShadowClassifier(n_qubits=n_qubits)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_dl:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val).item()
            _, predicted_val = torch.max(outputs_val, 1)
            val_acc = (predicted_val == y_val).sum().item() / y_val.size(0)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {total_loss/total:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

    return model, val_acc


# --- Main Execution ---

if __name__ == "__main__":
    N_QUBITS = 12
    G_POINTS = [0.5, 1.5]  # One in ordered, one in disordered
    N_SNAPSHOTS = 20000  # Per phase

    print(f"--- Classical Shadow Phase Classification ---")
    print(f"System: TFIM Chain, N={N_QUBITS}")
    print(f"Phases: g={G_POINTS} (Order vs Disorder)")

    # Generate Data
    t0 = time.time()
    X, y = generate_shadow_data(N_QUBITS, G_POINTS, N_SNAPSHOTS)
    print(f"Data generation time: {time.time()-t0:.2f}s")
    print(f"Feature shape: {X.shape}")  # Expect 2 * N_QUBITS

    # Train
    model, final_acc = train_model(X, y)

    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
