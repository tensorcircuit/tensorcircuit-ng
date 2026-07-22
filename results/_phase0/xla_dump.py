"""XLA buffer-assignment dump probe (correction Task A2). Best-effort.

Sets ``--xla_dump_to`` BEFORE importing jax (worker pattern), compiles the
n=24/d=10/default expectation executable, and enumerates whether the XLA dump yields any
parseable buffer-assignment / allocation / liveness artifact. If it does, the audit can be
enriched; if not, a structured blocker is recorded (allocation_status stays ``UNKNOWN`` --
rereview §4.1/4.3: the in-process ``serialized_buffer_assignment_proto`` is empty on GPU,
so the dump is the alternative, and a negative result is a determined UNKNOWN, not a gap).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
     python results/_phase0/xla_dump.py
"""

from __future__ import annotations

import json
import os
import sys

DUMP_DIR = "results/phase0/c1_xla_dump/n24_d10_default"
HLO_DIR = "results/phase0/c1_optimized_hlo"
OUT_JSON = "results/phase0/c1_xla_dump/n24_d10_default_summary.json"


def _file_signature(path):
    """Cheap content signature markers for a dump file (first non-empty line tokens)."""
    try:
        with open(path) as fh:
            head = fh.read(4096)
    except OSError:
        return ""
    return head


def run(n=24, depth=10):
    # XLA_FLAGS MUST be set before the first `import jax` -- do it here, then import lazily.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    prev = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (prev + f" --xla_dump_to={DUMP_DIR}").strip()
    if os.path.exists(DUMP_DIR):
        # start clean so the enumeration reflects THIS run only
        import shutil

        shutil.rmtree(DUMP_DIR, ignore_errors=True)
    os.makedirs(DUMP_DIR, exist_ok=True)

    import jax  # noqa: E402  (lazy: XLA_FLAGS honored at this import)
    import jax.numpy as jnp  # noqa: E402
    import tensorcircuit as tc  # noqa: E402

    from results._phase0.circuits import expectation_fn  # noqa: E402

    tc.set_backend("jax")
    theta = jnp.full(depth * n, 0.7, dtype=jnp.float32)
    f = expectation_fn(n, depth)
    compiled = f.lower(theta).compile()
    jax.block_until_ready(compiled(theta))

    # enumerate what XLA actually wrote
    files = []
    for root, _, fs in os.walk(DUMP_DIR):
        for fn in fs:
            files.append(os.path.relpath(os.path.join(root, fn), DUMP_DIR))
    files.sort()
    ba_files = [
        f
        for f in files
        if "buffer_assignment" in f.lower() or "buffer-assignment" in f.lower()
    ]
    alloc_files = [f for f in files if "allocation" in f.lower()]
    # also surface any file whose content mentions allocation/buffer-assignment text markers
    ba_marker_files = []
    for f in files:
        p = os.path.join(DUMP_DIR, f)
        if os.path.getsize(p) > 2_000_000:
            continue
        head = _file_signature(p)
        if "BufferAssignment" in head or "buffer_assignment" in head.lower():
            ba_marker_files.append(f)

    summary = {
        "n": n,
        "depth": depth,
        "fusion": "default",
        "dump_dir": DUMP_DIR,
        "xla_flags": os.environ["XLA_FLAGS"],
        "file_count": len(files),
        "file_names_sample": files[:25],
        "buffer_assignment_files": ba_files,
        "allocation_files": alloc_files,
        "buffer_assignment_marker_files": ba_marker_files,
        "has_parseable_buffer_assignment": bool(ba_files or ba_marker_files),
        "verdict": (
            "DUMP_HAS_BUFFER_ASSIGNMENT"
            if (ba_files or ba_marker_files)
            else "DUMP_NO_BUFFER_ASSIGNMENT_BLOCKER"
        ),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # subprocess worker entry (kept for parity with the c1 worker pattern)
        print(json.dumps(run()))
    else:
        print(json.dumps(run(), indent=2))
