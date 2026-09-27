# TNALG: Fixed-Shape Tensor-Network Algorithms

> Experimental, JAX-only, and rapidly evolving. Use at your own risk.

TNALG is a specialized TensorCircuit-NG subsystem for fixed-shape tensor-network algorithms. It provides functional MPS and MPO states, static layout specifications, Abelian block-sparse representations, and JAX-compatible implementations of TEBD, one-site TDVP, and one-site DMRG.

TNALG is not intended to replace every existing MPS or tensor-network interface in TensorCircuit. It uses a different execution model: physical dimensions, bond dimensions, symmetry sectors, and algorithm schedules are prepared ahead of the numerical hot path, while tensor buffers, coefficients, and time steps remain dynamic JAX values.

## Relationship to the other MPS abstractions

TensorCircuit currently has several MPS/MPO-related interfaces with different purposes:

| Interface | Role | Main characteristics |
| --- | --- | --- |
| `QuVector` / `QuOperator` | Generic tensor-network value objects | Represent arbitrary tensor-network graphs as vectors or operators; support tensor-network algebra and interoperability with `Circuit` and external packages. |
| `MPSCircuit` | Gate-oriented MPS circuit simulator | A mutable, circuit-like object for applying gates, truncating bonds, measuring, sampling, and evaluating observables. |
| `tnalg.MPSState` / `tnalg.MPOState` | Fixed-shape algorithm states | JAX PyTrees with explicit static specifications, designed for JIT, autodiff, block-sparse symmetry, and repeated TEBD/TDVP/DMRG execution. |

`QuVector` can represent an MPS, but it is a general tensor-network graph abstraction and is not required to have a linear-chain MPS topology. `MPSCircuit` and TNALG can represent related physical states, but they have different state lifecycles, truncation behavior, and execution models. They should not be treated as interchangeable live views of the same state.

The word “TEBD” also appears in two related contexts. `MPSCircuit` is an MPS-based gate simulator, whereas `tnalg` implements Hamiltonian time evolution with a fixed Suzuki–Trotter schedule. For new fixed-shape TEBD, TDVP, or DMRG workflows, use the TNALG APIs. For circuit-style gate application, measurement, and sampling, use `MPSCircuit` or another circuit class.

## Interoperability

TNALG provides explicit boundary conversions such as `as_mps`, `to_tn_mps`, `to_mpscircuit`, `as_mpo`, and `to_tn_mpo`. These conversions create independent objects; they do not establish shared mutable state or synchronization between the source and destination.

Conversions across these boundaries may also change the representation. In particular, exporting a symmetric TNALG state to a legacy dense MPS/MPO object requires explicit densification, and exporting to `MPSCircuit` requires uniform physical dimensions. Check the target format, axis order, truncation budget, basis order, and symmetry metadata before using a converted object.

`QuVector`/`QuOperator` and TeNPy conversions are maintained in the broader TensorCircuit quantum/tensor-network interoperability layer. A `QuOperator` is not automatically a canonical MPO, and a `QuVector` is not automatically a TNALG `MPSState`. Use explicit chain-aware conversions when moving between these representations.

## JAX-only execution model

TNALG is currently implemented directly with JAX primitives, including JAX PyTrees, `jax.jit`, `jax.lax`, `jax.vmap`, JAX autodiff, and JAX-specific linear-algebra helpers. It therefore requires JAX and should be regarded as a JAX-native subsystem.

Changing TensorCircuit's global backend does not make TNALG backend-agnostic. TensorFlow, PyTorch, NumPy, and other TensorCircuit backends are not currently supported by the TNALG algorithms.

## Symmetry-aware accuracy trade-off

The symmetry-aware algorithms are not guaranteed to be especially accurate in every problem. They require the bond-dimension fraction allocated to each symmetry sector to be specified in advance, and this fixed allocation may not match the manifold containing the target state. This is a trade-off made to retain a fixed structure for JAX just-in-time compilation; how to resolve the resulting tension between compilation-friendly static layouts and adaptive sector allocation remains an open issue.

## Development status and risk

This module has been developed largely with AI assistance and is still under active development. APIs, tensor layouts, algorithm schedules, numerical behavior, and interoperability boundaries may change quickly. Backward compatibility is not guaranteed at this stage.

Treat TNALG as an experimental research component. Pin the repository revision used for an experiment, validate numerical results against an independent reference when possible, and avoid relying on undocumented behavior in production workflows.
