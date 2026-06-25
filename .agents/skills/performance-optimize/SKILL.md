---
name: performance-optimize
description: Analyzes and refactors TensorCircuit-NG code to achieve peak time and memory performance. It enforces advanced JAX vectorization, intelligent JIT staging, optimal tensor network contraction, and memory-efficient autodiff strategies.
allowed-tools: Bash, Read, Grep, Glob, Write
---

When tasked with reviewing, profiling, or optimizing a TensorCircuit-NG (TC-NG) implementation, you act as a Principal HPC Quantum Software Engineer. Your goal is to optimize JIT compilation time, execution time, and peak memory, but you must do so scientifically—relying on empirical evidence rather than blindly applying rules.

### 1. Initial Profiling & Bottleneck Diagnosis
- **Analyze the Script**: Read the target Python file. Identify if the primary bottleneck is JIT staging time, execution time, or memory exhaustion (OOM) during forward passes or gradient computations.
- **Hardware Context**: Determine if the script is scaling to 20+ qubits or using large batch dimensions.

### 2. The TC-NG Performance Checklist (Hypothesis Generation)
Consider the following optimizations, but treat them as hypotheses to be tested:
- **Vectorization**: Replace manual `for` loops with `tc.backend.vmap` (or `jax.vmap`).
- **JIT Compilation**: Wrap performance-critical functions with `tc.backend.jit` (Ensure tensor-in, tensor-out). Better ensure JIT is placed in the most outside part possible (e.g., the entire optimization step). *Trade-off: JIT staging takes time; it is a net negative if the function is only executed once.*
- **JIT Reuse Discipline**: Keep jitted function input shapes, dtypes, Python container structure, and static arguments stable across calls. Put structural parameters such as qubit count, depth, topology, ansatz layout, solver options, and non-tensor flags in `static_argnums` when needed. Set backend and dtype before tracing, and avoid changing dtype or shape inside the training loop, otherwise the backend may retrace/recompile instead of reusing the compiled executable.
- **JIT Boundary for Gradients and Optimizers**: Prefer jitting the whole optimizer step instead of returning raw gradients from a jitted `value_and_grad` call. A robust pattern is:
  ```python
  @jax.jit
  def train_step(params, opt_state):
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss
  ```
  Avoid hot paths like `loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)` followed by a Python-side optimizer update unless profiling confirms compilation is healthy. In block-structured TensorCircuit workloads, returning a full flat gradient can force XLA to materialize the gradient as a live-out root. Reverse-mode of `params[offset:offset+k]` often lowers to many local-gradient `pad`/`add` fragments; if the full flat gradient is returned, XLA CPU fusion may combine these fragments into a single huge root fusion and make compile time explode. If gradient diagnostics are needed, return scalar summaries such as `grad_norm`, a few selected slices, or structured chunks at low frequency rather than the full raw gradient on the main training path.
- **Structured Parameters Over Flat Vectors**: Preserve the natural parameter structure when possible. Use PyTrees, tuples/lists of gate parameter blocks, or arrays shaped like `(n_gates, params_per_gate)` instead of a single long `params` vector with many manual `params[offset:offset+k]` slices. Flat vectors are acceptable at I/O boundaries or for SciPy-style APIs, but they can be unfriendly to autodiff and XLA because slice transposes assemble gradients through `pad`/`add` into a large dense vector. If flattening is unavoidable, benchmark both the flat form and a structured/PyTree form and inspect whether the compiled HLO contains a very large root `*_fusion` for gradient assembly.
- **XLA Fusion Pathology Check**: If first-call JIT compile time is far worse than expected while tracing/lowering are quick, test whether the problem is XLA fusion/codegen rather than TensorCircuit or JAX tracing. Useful probes include `JAX_LOG_COMPILES=1`, `XLA_FLAGS="--xla_dump_to=<tmp> --xla_dump_hlo_as_text"`, and a controlled run with `XLA_FLAGS=--xla_disable_hlo_passes=fusion`. If disabling fusion cuts compile time from minutes to seconds, look for oversized root fusions in `cpu_after_optimizations.txt`, especially fusions that assemble live-out gradients, Jacobians, residuals, or scatter/pad outputs. Treat `--xla_disable_hlo_passes=fusion` as a diagnostic or last-resort workaround: it may reduce compile time but can reduce runtime performance, so benchmark end-to-end before keeping it.
- **Staging Awareness**: Single-qubit gates can have longer JIT staging times than two-qubit gates. Avoid unnecessarily unrolling them.
- **Static Setup vs. Numeric Loop**: Move static work out of the hot path: Hamiltonian construction, observable/MPO/sparse-matrix construction, circuit topology selection, random graph generation, contraction optimizer setup, and shape-dependent preprocessing should happen once before the jitted numeric kernel or optimization loop.
- **Scan for Depth**: Use `tc.backend.scan` (or `jax.lax.scan`) for deep circuits with repeating structures instead of Python loops. When using scan, ensure the state is allowed in memory for the given number of qubits.
- **Host-Device Transfer Hygiene**: Avoid `tc.backend.numpy(...)`, `.item()`, Python `float(...)`, printing tensors, or Python branching on tensor values inside jitted or repeatedly executed kernels. Convert to host values only at logging, checkpointing, or benchmark boundaries.
- **Dtype Policy**: Use `complex64`/`float32` unless the physics or gradient check requires `complex128`/`float64`. Set dtype once before building long-lived constants and before JIT tracing so constants do not force recompilation or unintended casts.
- **Avoid Full State Instantiation**: For large qubit counts, ensure switches like `reuse=False` or `allow_state=False` in methods like `expectation`, `expectation_ps`, or `sample` are turned off to not form the full state.
- **Memory vs. Compute (Checkpointing)**: Nest `jax.checkpoint` within `scan` loops for deep circuit gradients. *Trade-off: This strictly trades increased forward-pass computation time for drastically reduced memory usage. Only apply if memory is the actual bottleneck.*
- **Advanced Contractor / Contract Strategy**: Prefer `OMECO` for large unsliced contraction-path search when available; in this repository it is typically much faster than running a full cotengra hyper-optimization over the same unsliced problem. TensorCircuit can auto-wrap `omeco.TreeSA` and `omeco.GreedyMethod` passed via `tc.set_contractor("custom", optimizer=..., preprocessing=True)`. Start from a minimal OMECO setup like:
  ```python
  import omeco
  import tensorcircuit as tc

  score = omeco.ScoreFunction(
      tc_weight=1.0,
      sc_weight=0.0,
      rw_weight=64.0,
      sc_target=20.0,
  )
  opt = omeco.TreeSA(
      ntrials=32,
      niters=32,
      betas=[0.5, 1.0, 2.0, 4.0, 8.0],
      score=score,
  )
  tc.set_contractor("custom", optimizer=opt, preprocessing=True)
  ```
  If slicing is required, use a hybrid strategy instead of pure cotengra search: first let OMECO find an unsliced path/tree, then hand that tree to cotengra for `slice_` and optional subtree reconfiguration. This matches the workflow in `develop/distributed_intermediate_contraction/distributed_intermediate_contraction_demo.py`, where OMECO seeds the unsliced tree and cotengra only handles slicing/post-processing. The current interface is not an end-to-end distributed OMECO planner, so do not promise native distributed support from OMECO alone.
  ```python
  import cotengra as ctg
  import omeco
  import tensorcircuit as tc

  ome_opt = tc.cons.OMEOptimizer(
      omeco.TreeSA(ntrials=32, niters=32, betas=[0.5, 1.0, 2.0, 4.0, 8.0])
  )
  path = ome_opt(input_sets, output_set, size_dict)
  tree = ctg.ContractionTree.from_path(input_sets, output_set, size_dict, path=path)
  tree.slice_(target_slices=target_slices, minimize="flops")
  tree = tree.subtree_reconfigure_forest(
      num_trees=8,
      num_restarts=16,
      subtree_weight_what=("size",),
  )
  ```
  If the target code only exposes a plain `tc.set_contractor(...)` hook and cannot manipulate a `ContractionTree` or cached path, fall back to tuned cotengra rather than forcing a brittle hybrid. In that fallback case, use a snippet like:
  ```python
  import cotengra as ctg
  import tensorcircuit as tc

  opt = ctg.ReusableHyperOptimizer(
      methods=["greedy", "kahypar"],
      parallel="ray",
      minimize="flops",
      max_time=120,
      max_repeats=1024,
      progbar=True,
  )
  tc.set_contractor("custom", optimizer=opt, preprocessing=True)
  ```
- **Contraction Path Assumption**: In typical jitted TensorCircuit workloads, contraction-path search is expected to be staged/reused rather than repeatedly executed in the hot path. Do not assume contraction-path search is the repeated bottleneck inside a jitted function unless profiling shows retracing, changing circuit structure, or changing static arguments.
- **Sparse/MPO/MVP Observables and ODEs**: Use Sparse Matrix, MPO, or matrix-free MVP representations for large system expectations instead of dense matrices or individual Pauli strings. Use APIs like `tc.templates.measurements.sparse_expectation`, `tc.templates.measurements.mpo_expectation`, and `tc.quantum.PauliStringSum2MVP`. For sparse Hamiltonians consisting of Pauli string sums, generate the matrix via `tc.quantum.PauliStringSum2COO`, which is significantly faster than manually using `kron`; if the Hamiltonian is too large to materialize, use `PauliStringSum2MVP` and compute `<psi|Hpsi>` with a tensor dot from the state and `Hpsi`. In JAX/Diffrax ODE vector fields, also prefer matrix-free Pauli-sum MVPs such as `H_mvp(y)` over sparse COO matmul such as `H @ y` when benchmarking supports it: adaptive ODE solve and reverse-mode tracing can otherwise carry large COO index constants through nested `jit` / `grad` / `eval_shape` graphs. In the challenge-suite digital-analog VQE case, replacing analog `COO @ y` calls with `PauliStringSum2MVP(y)` preserved the energy result while reducing end-to-end time by about an order of magnitude.
- **Backend Selection (PyTorch to JAX)**: Highly recommend moving from the PyTorch backend to JAX for significant performance gains (JIT, Vectorization). If the codebase is tightly integrated with the PyTorch ecosystem (e.g., using Torch optimizers or complex NN layers), use the JAX backend for the quantum kernel and bridge it using `tc.interfaces.torch_interface`. This keeps the quantum part fast on JAX while remaining end-to-end differentiable within the PyTorch computational graph.

### 3. Empirical Benchmarking & Trade-off Analysis
You MUST NOT assume an optimization is inherently better. 
- **A/B Testing**: Where computationally feasible, write a brief benchmarking snippet within the script (using `time` or `timeit`) to compare the original implementation against your refactored version on a scaled-down dummy input.
- **Evaluate the Trade-off**: Did `checkpoint` save enough memory to justify the time penalty? Did OMECO alone outperform tuned cotengra? If slicing was needed, did the hybrid `OMECO unsliced seed -> cotengra slice/reconfigure` workflow beat pure cotengra enough to justify the extra plumbing?
- **Revert on Regression**: If an "optimization" severely degrades overall performance for the specific use case, revert that specific change and document why.

### 4. Refactoring & Clean Execution
- **Apply Validated Changes**: Modify the target script with the optimizations that passed the trade-off analysis. Leave clear comments (e.g., `# Using scan + checkpoint here to prevent OOM during grad, accepting ~20% time overhead`).
- **Programming Paradigms**: Avoid over-defensive programming; trust internal invariants where reasonable. Use `try...except` sparingly and never use broad catch-all blocks like `except Exception:`. Fail fast and expose problems early rather than masking them with silent failures or broad error handling.
- **Dry Run**: Execute the final refactored script to ensure mathematically identical results to the original.

### 5. Output & Delivery
- **Save the Script**: Save the optimized script as `[original_name]_optimized.py` in the same directory as the original script.
- **Summary Report**: Conclude your task by providing a summary report containing:
  1. **Bottlenecks Identified**: What was the original issue?
  2. **Empirical Results & Trade-offs**: Which optimizations were tested, what were the benchmarked results (time/memory), and *why* specific techniques (like checkpointing) were kept or discarded based on the trade-off.
  3. **The Optimized Code**: Present the fully refactored, runnable, and highly optimized script.
