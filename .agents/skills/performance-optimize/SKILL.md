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
- **Problem Reformulation Before Micro-Optimization**: Before tuning individual gates, tensor contractions, or backend flags, ask whether the workload can be rewritten into a more favorable tensor program. In TC-NG, the biggest speedups often come from changing the mathematical representation rather than polishing the original one. Prefer sparse, MPO, matrix-free, light-cone, or direct-overlap formulations when they capture the same physics with less traced work or fewer intermediates.
- **Prefer the Most Native Representation That Matches the Physics**: When TC-NG already has a representation close to the real problem structure, use it directly instead of forcing everything through a generic statevector-style path. Examples include `tc.quantum.PauliStringSum2MVP`, `tc.quantum.PauliStringSum2COO`, `tc.templates.measurements.mpo_expectation`, direct MPS/MPO/qop contraction, `Circuit.post_select`, `enable_lightcone=True`, `QuditCircuit`, and built-in multi-qubit gates such as `cmz` or `su4`. These representations often beat more generic state-based formulations by reducing traced program size, dense-state formation, and intermediate tensor growth.
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
- **Move Static Physics/Data Out of the Hot Path**: Prebuild every object whose structure does not change across optimizer steps or repeated evaluations. This commonly includes sparse Hamiltonians, MPOs, matrix-free MVP callables, target bras/MPS objects, fixed gate tapes, bond lists, sampled Pauli structures, and reproducible shot-status tensors. The hot path should ideally depend only on trainable tensors plus small static metadata, which reduces retracing risk, avoids repeated setup, and makes benchmark timing easier to interpret.
- **Structured Parameters Over Flat Vectors**: Preserve the natural parameter structure when possible. Use PyTrees, tuples/lists of gate parameter blocks, or arrays shaped like `(n_gates, params_per_gate)` instead of a single long `params` vector with many manual `params[offset:offset+k]` slices. Flat vectors are acceptable at I/O boundaries or for SciPy-style APIs, but they can be unfriendly to autodiff and XLA because slice transposes assemble gradients through `pad`/`add` into a large dense vector. If flattening is unavoidable, benchmark both the flat form and a structured/PyTree form and inspect whether the compiled HLO contains a very large root `*_fusion` for gradient assembly.
- **Use Locality-Aware Evaluation Early**: For local observables or sparse sets of measured terms, first ask whether the computation admits light-cone reduction before investing effort in lower-level contraction tuning. In TC-NG, `enable_lightcone=True` can be an optimization strategy for observables generated by local circuits. Treat this as a benchmarkable hypothesis, not an automatic rule.
- **XLA Fusion Pathology Check**: If first-call JIT compile time is far worse than expected while tracing/lowering are quick, test whether the problem is XLA fusion/codegen rather than TensorCircuit or JAX tracing. Useful probes include `JAX_LOG_COMPILES=1`, `XLA_FLAGS="--xla_dump_to=<tmp> --xla_dump_hlo_as_text"`, and a controlled run with `XLA_FLAGS=--xla_disable_hlo_passes=fusion`. If disabling fusion cuts compile time from minutes to seconds, look for oversized root fusions in `cpu_after_optimizations.txt`, especially fusions that assemble live-out gradients, Jacobians, residuals, or scatter/pad outputs. Treat `--xla_disable_hlo_passes=fusion` as a diagnostic or last-resort workaround: it may reduce compile time but can reduce runtime performance, so benchmark end-to-end before keeping it.
- **GPU XLA Autotune Check**: For large JAX GPU tensor-network contractions or multi-GPU slicing runs, explicitly benchmark default GPU autotune against `XLA_FLAGS=--xla_gpu_autotune_level=0`, normally with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. This flag is a GPU/XLA behavior, not a CPU tuning knob. GPU autotune benchmarks/selects backend kernel algorithms and may retain algorithm/workspace resources, so it can increase compile memory, first-run memory, and first-device asymmetry. In tested TC-NG contraction workloads, disabling it reduced peak memory and first-call cost without steady-runtime loss, but keep it as an empirical A/B test rather than a blanket rule.
- **XLA Flag Combination Discipline**: In TC-NG, `gpu_memory_share(True)` already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so focus first on A/B testing `--xla_gpu_autotune_level=0`. Lowering `--xla_backend_optimization_level` may slightly change compile time but should not be expected to reduce contraction memory or improve steady runtime without measurement. Treat `--xla_disable_hlo_passes=fusion` as diagnostic only; on fixed-path GPU contraction-gradient tests it made compile and runtime worse without memory benefit. `--xla_gpu_enable_llvm_module_compilation_parallelism=true` is environment-dependent and can fail if the CUDA/XLA installation lacks the required parallel compilation provider.
- **Staging Awareness**: Single-qubit gates can have longer JIT staging times than two-qubit gates. Avoid unnecessarily unrolling them.
- **Static Setup vs. Numeric Loop**: Move static work out of the hot path: Hamiltonian construction, observables, path search, fixed tensors, and optimizer objects should be prepared once and reused. But do not mistake one-time setup savings for true steady-state speedups.
- **Sampling Path Choice Matters**: For large-qubit or large-shot sampling tasks, benchmark the sampling algorithm itself rather than assuming state construction is the dominant cost. If the task does not need an explicit final state and can use fixed-shape per-shot randomness, prefer benchmarking `Circuit.sample(batch=..., allow_state=False)`. Treat this as a large-system recommendation, not a blanket rule for small circuits. For reproducible benchmarking, generate the shot-status tensor once outside the timed or differentiated hot path and reuse its shape across calls.
- **Host-Device Transfer Hygiene**: Avoid `tc.backend.numpy(...)`, `.item()`, Python `float(...)`, printing tensors, or Python branching on tensor values inside jitted or repeatedly executed kernels. Convert to host values only at logging, checkpointing, or benchmark boundaries.
- **Dtype Policy**: Use `complex64`/`float32` unless the physics or gradient check requires `complex128`/`float64`. Set dtype once before building long-lived constants and before JIT tracing so constants do not force recompilation or unintended casts.
- **Avoid Full State Instantiation**: For large qubit counts, treat dense-state formation as a benchmark hypothesis rather than a default recommendation. Test whether switches such as `reuse=False` or `allow_state=False` in methods like `expectation`, `expectation_ps`, or `sample` are causing unnecessary state materialization on the concrete workload. Keep or change these settings only after benchmarking, since for smaller problems or different contraction regimes the full-state path can still be faster.
- **Control Flow, Memory, and Checkpointing**: For deep or repeated blocks, consider scanning over the evolving quantum state to reduce graph size. A common pattern is to make one layer/block consume a state and parameters and return the next state, for example:
  ```python
  import jax

  def layer_step(state, layer_params):
      c = tc.Circuit(nqubits, inputs=state)
      # apply one repeated layer/block here
      new_state = c.state()
      return new_state, None

  final_state, _ = jax.lax.scan(layer_step, init_state, params)
  ```
  When reverse-mode memory is the actual bottleneck, consider nesting `jax.checkpoint` / `jax.remat` around the per-layer function inside the `scan` loop. *Trade-off: `scan` can reduce graph size and compile burden, while rematerialization trades increased forward-pass computation time for reduced memory usage; neither should be applied by default without benchmarking.*
- **Compile-Time Mode vs Throughput Mode**: For repeated-layer circuits, do not treat `scan`, full unrolling, and partial unrolling as interchangeable. `scan` often wins by reducing traced code size and first-call compilation burden, while a manually unrolled circuit can still win on steady-state runtime after compilation. Report these as different modes when the trade-off matters, and tie the recommendation to expected reuse count.
- **Contraction Optimization (Cotengra/OMECO)**: For large tensor networks, optimize the contraction order. Prefer starting from OMECO as the default first recommendation, especially for large algebraic or hyperedge workloads, because path search is often much faster than cotengra hyper-optimization while still producing a strong usable contraction order. Minimal starting points are:
  ```python
  import tensorcircuit as tc

  tc.set_contractor("omeco")
  tc.set_contractor("omeco-16-32")  # ntrials=16, niters=32
  ```
  and for a cotengra baseline:
  ```python
  import cotengra as ctg
  import tensorcircuit as tc

  opt = ctg.ReusableHyperOptimizer(
      methods=["greedy", "kahypar"],
      minimize="flops",
      max_time=30,
      max_repeats=128,
      progbar=True,
  )
  tc.set_contractor("custom", optimizer=opt, preprocessing=True)
  ```
  Treat the optimizer budget as problem-dependent: the right search depth, retries, or time budget should be benchmarked on the concrete workload rather than fixed by rule. When the resulting contraction is reused many times, a slower but stronger-searching optimizer can still win overall, so compare OMECO against tuned cotengra when amortization is plausible.
- **Contraction Slicing / Reconfiguration Hybrid**: If exact contraction is still too expensive after choosing a good unsliced path, consider a two-stage workflow: first find a strong unsliced path quickly (for example with OMECO), then build a `cotengra.ContractionTree` from that path and let cotengra handle slicing and subtree reconfiguration. This can outperform pure cotengra search when the search space is large but the final sliced execution still benefits from cotengra's slicing heuristics. A schematic example is:
  ```python
  import cotengra as ctg

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
- **Contraction Tuning Is Not Always the First Lever**: If the network can first be reduced by light-cone reduction, sparse/MPO/MPS reformulation, direct overlap contraction, or a more native circuit/operator API, benchmark those changes before spending effort on path-search tuning. In many workloads, contraction optimization is the second-order gain after the program representation has already been improved.
- **Sparse/MPO/MVP Observables and ODEs**: Use Sparse Matrix, MPO, or matrix-free MVP representations for large system expectations instead of dense matrices or individual Pauli strings. Use APIs like `tc.templates.measurements.sparse_expectation`, `tc.templates.measurements.mpo_expectation`, and `tc.quantum.PauliStringSum2MVP`. For sparse Hamiltonians consisting of Pauli string sums, generate the matrix via `tc.quantum.PauliStringSum2COO`, which is significantly faster than manually using `kron`; if the Hamiltonian is too large to materialize, use `PauliStringSum2MVP` and compute `<psi|Hpsi>` with a tensor dot from the state and `Hpsi`. In JAX/Diffrax ODE vector fields, also prefer matrix-free Pauli-sum MVPs such as `H_mvp(y)` over sparse COO matmul such as `H @ y` when benchmarking supports it: adaptive ODE solve and reverse-mode tracing can otherwise carry large COO index constants through nested `jit` / `grad` / `eval_shape` graphs. In the challenge-suite digital-analog VQE case, replacing analog `COO @ y` calls with `PauliStringSum2MVP(y)` preserved the energy result while reducing end-to-end time by about an order of magnitude.
- **Backend Selection (PyTorch to JAX)**: Highly recommend moving from the PyTorch backend to JAX for significant performance gains (JIT, Vectorization). If the codebase is tightly integrated with the PyTorch ecosystem (e.g., using Torch optimizers or complex NN layers), use the JAX backend for the quantum kernel and bridge it using `tc.interfaces.torch_interface`. This keeps the quantum part fast on JAX while remaining end-to-end differentiable within the PyTorch computational graph.

### 3. Empirical Benchmarking & Trade-off Analysis
You MUST NOT assume an optimization is inherently better. 
- **A/B Testing**: Where computationally feasible, write a brief benchmarking snippet within the script (using `time` or `timeit`) to compare the original implementation against your refactored version on a scaled-down dummy input.
- **Benchmark the Representation Choice**: If you changed the mathematical form of the computation—such as local-term loop -> MVP/MPO, dense target -> direct overlap contraction, generic expectation -> light-cone evaluation, or state-based sampling -> direct tensor-network sampling—treat that rewrite as the primary benchmarked change and document it explicitly before discussing lower-level tuning.
- **Evaluate the Trade-off**: Did `checkpoint` save enough memory to justify the time penalty? Did OMECO alone outperform tuned cotengra? If slicing was needed, did the hybrid `OMECO unsliced seed -> cotengra slice/reconfigure` workflow beat pure cotengra enough to justify the extra plumbing?
- **Separate Staging from Steady State**: Whenever JIT, contraction-path search, or static-object construction is involved, report one-time setup/warmup cost separately from repeated execution cost. For workloads run many times, estimate the break-even point rather than reporting only a single aggregated runtime.
- **Report GPU Runtime Phases Separately**: For large JAX GPU jobs, record memory and time for compile/lower, first execution, second or steady execution, and post-cache-clear state separately. Default GPU autotune can leave retained first-device memory that is not released by `jax.clear_caches()`, so a single peak number can hide whether the memory came from the executable/runtime buffers or from autotune/cache overhead.
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
