"""C1 measurement with compile/runtime memory split, optimized HLO + buffer assignment (review §5.2/5.3).

Two upstream handoffs from Task 3 (results/_phase0_circuits.py):
- ``expectation_fn(n, depth)`` -> ``f(theta)`` (jax.jit) whose output is complex64 (apply ``.real`` for a float).
  ``theta`` length must be >= ``depth*n``. Task 3 confirmed the parameterized circuit defeats XLA constant
  folding (stablehlo carries ``%arg0``).

CRITICAL gotcha (jax import ordering): ``XLA_FLAGS`` (e.g. ``--xla_disable_hlo_passes=fusion``) is read at
``import jax`` time. The brief's ``measure_case`` called ``_set_xla_flags()`` INSIDE the function, but this
module imports jax at top -> that call is INEFFECTIVE (this exact bug was hit earlier this session). The robust
pattern (mirrors results/_phase0_fusion_window_probe.worker_main) is a worker entry that sets
``os.environ["XLA_FLAGS"]`` from argv BEFORE importing jax, then runs the measurement. Structure here:
- ``measure_case(...)`` does NOT rely on post-import flag setting; it treats ``disable_fusion`` as a LABEL
  (records which arm + drives artifact filenames).
- ``worker_main(argv)`` parses ``--n --depth --disable-fusion --theta-seed``, sets ``XLA_FLAGS`` if
  ``disable_fusion`` BEFORE ``import jax``, then imports the measurement module + runs ``measure_case``,
  emitting one JSON line via ``worker_emit``.

Step-1 probe result (jax 0.6.2): ``compiled.memory_analysis()`` IS available, returning a
``jaxlib._jax.CompiledMemoryStats`` object. Its attributes are SCALAR bytes (NOT the ``a_sizes``/``temp_sizes``
lists the brief assumed). Real attributes: ``alias_size_in_bytes``, ``argument_size_in_bytes``,
``generated_code_size_in_bytes``, ``output_size_in_bytes``, ``temp_size_in_bytes`` (plus ``host_*`` variants
and ``serialized_buffer_assignment_proto`` -> bytes, length recorded). The brief's ``list(m.a_sizes)``
would have raised ``TypeError: 'NoneType' object is not iterable``; this module records the real scalars
instead (documented deviation). ``compiled.as_text()`` works for optimized HLO (no fallback needed).
"""

from __future__ import annotations
import argparse
import os
import sys

# NOTE: jax is NOT imported at module top so the worker can set XLA_FLAGS first. ``measure_case`` does a
# lazy ``import jax`` / ``import tensorcircuit`` inside the function body.

OUT_DIR = "results/phase0"


def _record_memory_analysis(compiled):
    """Prefer ``compiled.memory_analysis()`` (jax 0.6.2 has it). Returns a JSON-serializable dict or None.

    Real attribute names (Step-1 probe; the brief's ``a_sizes``/``temp_sizes`` do NOT exist on this object):
    scalar ``*_in_bytes`` counters, ``host_*`` mirrors, and a serialized buffer-assignment proto (bytes).
    """
    if not hasattr(compiled, "memory_analysis"):
        return None
    try:
        m = compiled.memory_analysis()
    except Exception as e:  # pragma: no cover - defensive
        return {"error": repr(e)[:200]}
    if m is None:
        return None
    scalars = {}
    for name in (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    ):
        scalars[name] = int(getattr(m, name, 0))
    try:
        proto = m.serialized_buffer_assignment_proto
        scalars["serialized_buffer_assignment_proto_len"] = len(proto) if proto else 0
    except Exception:  # pragma: no cover - defensive
        scalars["serialized_buffer_assignment_proto_len"] = -1
    return scalars


def measure_case(n, depth, theta_seed=0.7, disable_fusion=False, repeats=3):
    """Compile/runtime memory split for the parameterized C1 circuit.

    ``disable_fusion`` is a LABEL only (filenames + result fields); it does NOT mutate ``XLA_FLAGS`` here
    because jax is already imported by the time this function runs. The worker entry sets the flag before
    ``import jax`` for the no-fusion arm.

    Returns a dict with ``compile_peak_B`` (peak_bytes_in_use after first compile+exec, with compile
    artifacts resident) and ``runtime_peak_B`` (max bytes_in_use across ``repeats`` steady-state execs).
    """
    import jax  # lazy: lets worker_main set XLA_FLAGS before jax import in no-fusion arm
    import jax.numpy as jnp
    import tensorcircuit as tc  # noqa: F401  (tc.set_backend in expectation_fn's module)

    from results._phase0_circuits import expectation_fn

    tc.set_backend("jax")
    theta = jnp.full(depth * n, theta_seed, dtype=jnp.float32)
    f = expectation_fn(n, depth)
    lowered = f.lower(theta)

    dev = jax.local_devices()[0]
    _ = dev.memory_stats()  # touch
    compile_peak_before = int(dev.memory_stats().get("bytes_in_use", 0))
    compiled = lowered.compile()
    # first exec compiles + leaves compile artifacts resident
    jax.block_until_ready(compiled(theta))
    compile_peak = int(dev.memory_stats().get("peak_bytes_in_use", 0))

    ma = _record_memory_analysis(compiled)

    # steady-state runtime peak: exec `repeats` times; jax has no per-window reset, so report the
    # delta-driven peak by comparing bytes_in_use before/after a tight exec loop.
    runtime_peaks = []
    for _ in range(repeats):
        b0 = int(dev.memory_stats().get("bytes_in_use", 0))
        jax.block_until_ready(compiled(theta))
        b1 = int(dev.memory_stats().get("bytes_in_use", 0))
        runtime_peaks.append(max(b0, b1))
    runtime_peak = max(runtime_peaks)

    fm = "nofusion" if disable_fusion else "default"
    hlo_path = f"{OUT_DIR}/c1_optimized_hlo/n{n}_d{depth}_exp_{fm}.hlo"
    os.makedirs(os.path.dirname(hlo_path), exist_ok=True)
    hlo_text = None
    try:
        hlo_text = compiled.as_text()
    except Exception as e:  # pragma: no cover - fallback per task brief
        hlo_text = (
            "### as_text() raised; fallback str(compiled.compiler_ir(dialect='stablehlo')):\n"
            + f"# as_text error: {repr(e)[:200]}\n"
            + str(compiled.compiler_ir(dialect="stablehlo"))
        )
    with open(hlo_path, "w") as fh:
        fh.write(hlo_text or "")

    ba_path = f"{OUT_DIR}/c1_buffer_assignment/n{n}_d{depth}_exp_{fm}.txt"
    os.makedirs(os.path.dirname(ba_path), exist_ok=True)
    ba_header = (
        "source: "
        + (
            "memory_analysis"
            if ma
            else "xla_dump_to (set XLA_FLAGS=--xla_dump_to externally)"
        )
        + "\n"
    )
    with open(ba_path, "w") as fh:
        fh.write(ba_header)
        fh.write(repr(ma) + "\n")

    return {
        "n": n,
        "depth": depth,
        "disable_fusion": bool(disable_fusion),
        "compile_peak_B": compile_peak,
        "compile_peak_before_B": compile_peak_before,
        "runtime_peak_B": runtime_peak,
        "runtime_peaks_B": runtime_peaks,
        "memory_analysis": ma,
        "hlo_path": hlo_path,
        "buffer_assignment_path": ba_path,
        "full_state_bytes": (2**n) * 8,
    }


def worker_main(argv):
    """Single (n, depth, disable_fusion, theta_seed) measurement. Prints one JSON line.

    Mirrors results/_phase0_fusion_window_probe.worker_main: sets XLA_FLAGS BEFORE ``import jax`` so the
    no-fusion arm actually disables fusion (the brief's in-function ``_set_xla_flags`` is a no-op once jax
    is imported).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--disable-fusion", type=int, default=0)
    ap.add_argument("--theta-seed", type=float, default=0.7)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)

    if a.disable_fusion:
        prev = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (prev + " --xla_disable_hlo_passes=fusion").strip()

    # late imports so the XLA_FLAGS set above is honored at jax init
    from results._phase0_common import worker_emit
    from results._phase0_c1 import measure_case

    try:
        result = measure_case(
            a.n,
            a.depth,
            theta_seed=a.theta_seed,
            disable_fusion=bool(a.disable_fusion),
            repeats=a.repeats,
        )
        result["outcome"] = "run"
        worker_emit(result)
    except Exception as e:
        worker_emit({"outcome": "crash", "error": repr(e)[:300]})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(
        description="C1 compile/runtime memory split (Task 4)."
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--disable-fusion", type=int, default=0)
    ap.add_argument("--theta-seed", type=float, default=0.7)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    # in-process default arm (no XLA_FLAGS mutation); for no-fusion arm invoke via `worker`.
    result = measure_case(
        a.n,
        a.depth,
        theta_seed=a.theta_seed,
        disable_fusion=bool(a.disable_fusion),
        repeats=a.repeats,
    )
    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
