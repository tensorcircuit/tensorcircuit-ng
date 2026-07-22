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
import csv
import json
import os
import re
import sys

# NOTE: jax is NOT imported at module top so the worker can set XLA_FLAGS first. ``measure_case`` does a
# lazy ``import jax`` / ``import tensorcircuit`` inside the function body.

OUT_DIR = "results/phase0"
JUDGMENT_JSON_PATH = f"{OUT_DIR}/c1_judgment.json"
AB_CSV_PATH = f"{OUT_DIR}/c1_default_vs_nofusion.csv"


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

    Returns a dict with:
    - ``compile_peak_B``: ``peak_bytes_in_use`` read after ``.compile()`` + first exec. CUMULATIVE — it
      includes the first-exec runtime temp — so it is NOT a clean compile-only figure.
    - ``runtime_peak_B``: the steady-state per-exec runtime materialization peak, taken as
      ``compiled.memory_analysis().temp_size_in_bytes`` (the XLA-computed max temp the compiled program
      allocates EVERY execution). This is the meaningful, attributable runtime metric.
    - ``post_exec_resident_B`` (diagnostic, NOT the peak): max ``bytes_in_use`` sampled before/after each
      exec. Both samples land when execution is NOT running, so this is the resident arg/output bucket
      left AFTER the in-exec temp is freed.
    """
    import jax  # lazy: lets worker_main set XLA_FLAGS before jax import in no-fusion arm
    import jax.numpy as jnp
    import tensorcircuit as tc  # noqa: F401  (tc.set_backend in expectation_fn's module)

    from results._phase0_circuits import expectation_fn

    tc.set_backend("jax")
    backend = jax.default_backend()
    theta = jnp.full(depth * n, theta_seed, dtype=jnp.float32)
    f = expectation_fn(n, depth)
    lowered = f.lower(theta)

    dev = jax.local_devices()[0]
    _ = dev.memory_stats()  # touch
    compile_peak_before = int(dev.memory_stats().get("bytes_in_use", 0))
    compiled = lowered.compile()
    # first exec compiles + leaves compile artifacts resident
    jax.block_until_ready(compiled(theta))
    # CUMULATIVE peak_bytes_in_use: includes the first-exec runtime temp, so NOT a clean compile-only figure.
    compile_peak = int(dev.memory_stats().get("peak_bytes_in_use", 0))

    ma = _record_memory_analysis(compiled)

    # Steady-state runtime peak = XLA-computed max temp the compiled program allocates EVERY execution
    # (the contraction scratch). This is the attributable per-exec runtime materialization peak. The prior
    # approach sampled bytes_in_use before/after block_until_ready, but both samples land when execution is
    # NOT running, so the transient in-exec temp was already freed -> it missed the runtime peak entirely
    # (review §5.2 sampling artifact).
    if isinstance(ma, dict) and "temp_size_in_bytes" in ma:
        runtime_peak = int(ma["temp_size_in_bytes"])
    else:  # pragma: no cover - defensive (memory_analysis unavailable)
        runtime_peak = 0

    # Diagnostic only: resident bytes after each exec (NOT the runtime peak). Both samples are taken when
    # execution is NOT running, so this reports the post-exec resident bucket, not the in-exec temp.
    post_exec_resident_samples = []
    for _ in range(repeats):
        b0 = int(dev.memory_stats().get("bytes_in_use", 0))
        jax.block_until_ready(compiled(theta))
        b1 = int(dev.memory_stats().get("bytes_in_use", 0))
        post_exec_resident_samples.append(max(b0, b1))
    post_exec_resident_B = max(post_exec_resident_samples)

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
        "backend": backend,
        "compile_peak_B": compile_peak,
        "compile_peak_before_B": compile_peak_before,
        "runtime_peak_B": runtime_peak,
        "post_exec_resident_B": post_exec_resident_B,
        "post_exec_resident_B_samples": post_exec_resident_samples,
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


def judge_c1(
    default_result,
    nofusion_result,
    repeats_results,
    materialized_buffer_bytes,
    optimized_hlo_has_materialized,
):
    """Six conditions (review §5.4). C1=YES needs ALL; any miss => FAIL or UNKNOWN.

    Condition 4 CORRECTION vs. the brief: the brief's ``(pd > 0) and (pn / pd >= 0.5)`` is inverted.
    Correct semantics for "the materialized temp is NOT XLA-eliminated": the temp PERSISTS in the
    fusion-ON (default) arm. Disabling fusion would make ``pn`` (fusion-OFF peak) roughly unchanged
    if fusion was irrelevant, or much LARGER if fusion was previously eliminating part of the temp.
    So "not eliminated" <=> the default-arm peak ``pd`` retains a large fraction of the no-fusion
    peak ``pn``: ``pd >= 0.5 * pn``. If fusion eliminated the intermediate, ``pn >> pd`` (disabling
    fusion reveals the eliminated materialization) -> ``pd < 0.5*pn`` -> condition fails. If fusion
    is irrelevant, ``pn ≈ pd`` -> condition passes.
    """
    state_bytes = default_result["full_state_bytes"]
    conds = {}
    # 1 dynamic params -> verified upstream in _phase0_circuits; caller passed the dynamic case.
    conds["1_dynamic_params"] = True
    # 2 optimized HLO shows a materialized contraction buffer
    conds["2_hlo_has_materialized_buffer"] = bool(optimized_hlo_has_materialized)
    # 3 materialized bytes >= 0.5 x full state
    conds["3_materialized_ge_half_state"] = (
        materialized_buffer_bytes >= 0.5 * state_bytes
    )
    # 4 NOT XLA-eliminated: default-arm peak retains >= half of no-fusion peak (see docstring)
    pd = default_result.get("runtime_peak_B", 0)
    pn = nofusion_result.get("runtime_peak_B", 0)
    conds["4_not_xla_eliminated"] = (pd > 0) and (pd >= 0.5 * pn)
    # 5 executable (caller ensures not crash/OOM); mark UNKNOWN if peak is 0
    conds["5_executable"] = pd > 0
    # 6 3x stable: runtime_peak consistent within 5% across the repeats arm
    peaks = [r.get("runtime_peak_B", 0) for r in repeats_results]
    if peaks:
        conds["6_repeat_stable"] = min(peaks) >= 0.95 * max(peaks)
    else:
        conds["6_repeat_stable"] = False
    if not conds["6_repeat_stable"]:
        return {
            "status": "UNKNOWN",
            "reason": "3x repeats unstable",
            "conditions": conds,
        }
    if not conds["5_executable"]:
        return {
            "status": "UNKNOWN",
            "reason": "not executable (peak 0)",
            "conditions": conds,
        }
    if not conds["3_materialized_ge_half_state"]:
        return {
            "status": "FAIL",
            "reason": (
                f"materialized {materialized_buffer_bytes} < 0.5x "
                f"state {state_bytes} (threshold {0.5 * state_bytes})"
            ),
            "conditions": conds,
        }
    if not conds["2_hlo_has_materialized_buffer"]:
        return {
            "status": "FAIL",
            "reason": "no materialized contraction buffer in optimized HLO",
            "conditions": conds,
        }
    if not conds["4_not_xla_eliminated"]:
        return {
            "status": "FAIL",
            "reason": (
                "evidence shows XLA eliminates it "
                "(default-arm peak < 0.5x no-fusion peak; fusion was removing it)"
            ),
            "conditions": conds,
        }
    return {"status": "PASS", "reason": "all 6 conditions met", "conditions": conds}


def _median_run(runs):
    """Pick the run whose ``runtime_peak_B`` is the median of the 3. Falls back to the first run.

    ``runs`` may be either a list of result dicts or a list of ``(config, result)`` pairs (the
    orchestrator pairs configs with results positionally; the pair form preserves ``theta_seed``).
    Returns just the result dict.
    """
    if not runs:
        return {}
    if runs and isinstance(runs[0], tuple):
        results = [r for _, r in runs]
    else:
        results = list(runs)
    if len(results) == 1:
        return results[0]
    peak_sorted = sorted(results, key=lambda r: r.get("runtime_peak_B", 0))
    return peak_sorted[len(peak_sorted) // 2]


def _median_theta_seed(runs):
    """Theta seed of the median run, recovered from the ``(config, result)`` pair form."""
    if not runs:
        return None
    if isinstance(runs[0], tuple):
        pairs = list(runs)
    else:
        return None
    if len(pairs) == 1:
        return pairs[0][0].get("theta_seed")
    peak_sorted = sorted(pairs, key=lambda pr: pr[1].get("runtime_peak_B", 0))
    return peak_sorted[len(peak_sorted) // 2][0].get("theta_seed")


def _build_c1_worker_argv(cfg):
    return [
        "--n",
        str(cfg["n"]),
        "--depth",
        str(cfg["depth"]),
        "--disable-fusion",
        str(cfg["disable_fusion"]),
        "--theta-seed",
        str(cfg["theta_seed"]),
    ]


def _append_csv_row(path, header, row):
    """Append ``row`` to ``path``; write ``header`` first if the file is new/empty."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(row)


def _update_judgment_json(path, key, payload):
    """Read-merge-write a dict keyed by ``key`` into the judgment JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[key] = payload
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2)


# --- Condition-2 HLO evidence (review §5.4): an INDEPENDENT source of truth ---
# Condition 2 must NOT reuse the memory metric (``materialized_buffer_bytes``), otherwise
# the six-condition gate collapses to five. Instead we parse ``compiled.as_text()`` for the
# largest materialized contraction buffer — the largest typed output shape of a
# ``__cublas$gemm`` custom-call or a raw ``dot_general`` op — convert element-count x
# bytes-per-element, and compare to 0.5 x full state. This reads only the optimized HLO text
# that ``measure_case`` already saves to ``results/phase0/c1_optimized_hlo/...hlo``.
_HLO_DTYPE_BYTES = {
    "f32": 4,
    "f64": 8,
    "bf16": 2,
    "f16": 2,
    "c64": 8,
    "c128": 16,
    "s8": 1,
    "s16": 2,
    "s32": 4,
    "s64": 8,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "pred": 1,
}
# Match the OUTPUT tuple of a ``__cublas$gemm`` custom-call, e.g.
#   %x = (c64[4096,4096]{1,0}, s8[33554432]{0}) custom-call(...)
# Captures the tuple body (no nested parens occur inside it).
_CUBLAS_TUPLE_RE = re.compile(r"=\s*\(([^)]*)\)\s+custom-call")
# Match a raw ``dot_general`` op with a single typed output, e.g.
#   %x = c64[4096,4096]{1,0} dot_general(...)
# Covers non-cuBLAS backends where the contraction is not lowered to a custom-call.
_DOT_GENERAL_SINGLE_RE = re.compile(
    r"=\s+([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}\s+dot_general\b"
)
# Match one typed tuple element: TYPE[dims]{layout}
_TYPED_ELEM_RE = re.compile(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}")


def _elem_bytes(dtype, dims_csv):
    """Element-count x bytes-per-element for an HLO shape. Returns 0 for unknown dtypes."""
    bytes_per = _HLO_DTYPE_BYTES.get(dtype)
    if not bytes_per:
        return 0
    count = 1
    for d in dims_csv.split(","):
        count *= int(d)
    return count * bytes_per


def largest_materialized_tensor_bytes_from_hlo(hlo_text):
    """Largest materialized contraction-buffer byte size found in optimized HLO text.

    Scans outputs of ``__cublas$gemm`` custom-calls (GPU) and raw ``dot_general`` ops
    (other backends), taking the max typed-output byte size. Returns 0 if no contraction
    op is found. This is the condition-2 evidence — INDEPENDENT of the runtime memory metric.
    """
    largest = 0
    for line in hlo_text.splitlines():
        if "__cublas$gemm" in line:
            m = _CUBLAS_TUPLE_RE.search(line)
            if not m:
                continue
            for elem in _TYPED_ELEM_RE.finditer(m.group(1)):
                largest = max(largest, _elem_bytes(elem.group(1), elem.group(2)))
        elif "dot_general" in line:
            m = _DOT_GENERAL_SINGLE_RE.search(line)
            if not m:
                continue
            largest = max(largest, _elem_bytes(m.group(1), m.group(2)))
    return largest


def run_c1_ab(n, depth, theta_seeds=(0.7, 0.8, 0.9)):
    """Run default + no-fusion arms (3× per arm, one per theta seed), then judge C1.

    Each arm is a fresh subprocess (XLA_FLAGS set in ``worker_main`` BEFORE ``import jax`` for the
    no-fusion arm), orchestrated via ``results._phase0_common.orchestrate``. The median run (by
    ``runtime_peak_B``) of each arm is fed to ``judge_c1``; the 3 default runs become
    ``repeats_results`` for the 3x-stable check (condition 6).

    Writes one row per (n, depth) to ``results/phase0/c1_default_vs_nofusion.csv`` and merges the
    judgment under key ``n{n}_d{depth}`` into ``results/phase0/c1_judgment.json``.
    """
    from results._phase0_common import orchestrate

    script_path = os.path.abspath(__file__)
    default_configs = [
        {"n": n, "depth": depth, "disable_fusion": 0, "theta_seed": s}
        for s in theta_seeds
    ]
    nofusion_configs = [
        {"n": n, "depth": depth, "disable_fusion": 1, "theta_seed": s}
        for s in theta_seeds
    ]

    default_rows = orchestrate(
        default_configs, _build_c1_worker_argv, script_path, timeout=1800
    )
    nofusion_rows = orchestrate(
        nofusion_configs, _build_c1_worker_argv, script_path, timeout=1800
    )

    default_pairs = [(r["config"], r["result"]) for r in default_rows if r.get("ok")]
    nofusion_pairs = [(r["config"], r["result"]) for r in nofusion_rows if r.get("ok")]
    default_runs = [r for _, r in default_pairs]
    nofusion_runs = [r for _, r in nofusion_pairs]

    default_median = _median_run(default_pairs)
    nofusion_median = _median_run(nofusion_pairs)
    default_median_theta = _median_theta_seed(default_pairs)
    nofusion_median_theta = _median_theta_seed(nofusion_pairs)

    full_state_bytes = (2**n) * 8
    pd_peak = int(default_median.get("runtime_peak_B", 0))
    pn_peak = int(nofusion_median.get("runtime_peak_B", 0))
    materialized_buffer_bytes = pd_peak

    # Condition-2 evidence (review §5.4): parse the DEFAULT arm's optimized HLO text for the
    # largest materialized contraction buffer — INDEPENDENT of the memory metric above.
    # ``measure_case`` already wrote ``compiled.as_text()`` to default_median["hlo_path"].
    largest_materialized_hlo_bytes = 0
    hlo_path = default_median.get("hlo_path")
    if hlo_path and os.path.exists(hlo_path):
        try:
            with open(hlo_path) as fh:
                hlo_text = fh.read()
            largest_materialized_hlo_bytes = largest_materialized_tensor_bytes_from_hlo(
                hlo_text
            )
        except OSError:  # pragma: no cover - defensive
            largest_materialized_hlo_bytes = 0
    optimized_hlo_has_materialized = (
        largest_materialized_hlo_bytes >= 0.5 * full_state_bytes
    )

    judgment = judge_c1(
        default_result=default_median,
        nofusion_result=nofusion_median,
        repeats_results=default_runs,
        materialized_buffer_bytes=materialized_buffer_bytes,
        optimized_hlo_has_materialized=optimized_hlo_has_materialized,
    )

    ratio = (pn_peak / pd_peak) if pd_peak > 0 else 0.0

    csv_header = [
        "n",
        "depth",
        "default_peak_B",
        "nofusion_peak_B",
        "ratio_nofusion_default",
        "default_median_theta_seed",
        "nofusion_median_theta_seed",
        "full_state_bytes",
        "c1_status",
    ]
    csv_row = [
        n,
        depth,
        pd_peak,
        pn_peak,
        f"{ratio:.4f}",
        default_median_theta,
        nofusion_median_theta,
        full_state_bytes,
        judgment["status"],
    ]
    _append_csv_row(AB_CSV_PATH, csv_header, csv_row)

    payload = {
        "n": n,
        "depth": depth,
        "default_peak_B": pd_peak,
        "nofusion_peak_B": pn_peak,
        "ratio_nofusion_default": ratio,
        "full_state_bytes": full_state_bytes,
        "largest_materialized_hlo_bytes": largest_materialized_hlo_bytes,
        "default_run_peaks_B": [int(r.get("runtime_peak_B", 0)) for r in default_runs],
        "nofusion_run_peaks_B": [
            int(r.get("runtime_peak_B", 0)) for r in nofusion_runs
        ],
        "default_failed": [
            {"config": r.get("config"), "outcome": r.get("outcome")}
            for r in default_rows
            if not r.get("ok")
        ],
        "nofusion_failed": [
            {"config": r.get("config"), "outcome": r.get("outcome")}
            for r in nofusion_rows
            if not r.get("ok")
        ],
        "judgment": judgment,
    }
    _update_judgment_json(JUDGMENT_JSON_PATH, f"n{n}_d{depth}", payload)

    return payload


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(
        description="C1 compile/runtime memory split (Task 4) + C1 judgment A/B (Task 5)."
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--disable-fusion", type=int, default=0)
    ap.add_argument("--theta-seed", type=float, default=0.7)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--ab",
        action="store_true",
        help="run run_c1_ab(n, depth): default+nofusion A/B (3x each) + judge_c1",
    )
    a = ap.parse_args()
    if a.ab:
        payload = run_c1_ab(a.n, a.depth)
        print(json.dumps(payload, indent=2))
        return
    # in-process default arm (no XLA_FLAGS mutation); for no-fusion arm invoke via `worker`.
    result = measure_case(
        a.n,
        a.depth,
        theta_seed=a.theta_seed,
        disable_fusion=bool(a.disable_fusion),
        repeats=a.repeats,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
