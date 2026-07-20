"""Aggregate bench_bf16_gpu CSV rows into a markdown report.

For each (backend, circuit, n): report bf16 vs complex64 peak-mem ratio, speedup,
and bf16 accuracy. For micro rows: report wall-time per backend.
"""
import argparse
import csv
from collections import defaultdict


def _gib(b: float) -> str:
    return f"{b / (1024 ** 3):.2f} GiB" if b else "-"


def aggregate(csv_path: str) -> str:
    e2e = defaultdict(dict)  # (backend, circuit, n) -> {dtype: row}
    micro = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("mode") == "micro":
                micro.append(r)
            else:
                e2e[(r["backend"], r["circuit"], int(r["n"]))][r["dtype"]] = r

    lines = ["# bf16 GPU benchmark results", ""]
    lines += ["## End-to-end (bf16 vs complex64)", ""]
    lines += [
        "| backend | circuit | n | c64 mem | bf16 mem | mem ratio | c64 s | bf16 s | speedup | bf16 max-abs-err |"
    ]
    lines += ["|---|---|---|---|---|---|---|---|---|---|"]
    for (backend, circuit, n), d in sorted(e2e.items()):
        c64, bf = d.get("complex64"), d.get("bf16")
        if not (c64 and bf):
            continue
        c64_mem = int(c64["peak_smi_bytes"] or 0)
        bf_mem = int(bf["peak_smi_bytes"] or 0)
        ratio = f"{c64_mem / bf_mem:.2f}x" if bf_mem else "-"
        c64_s = float(c64["wall_s"])
        bf_s = float(bf["wall_s"])
        speedup = f"{c64_s / bf_s:.2f}x" if bf_s else "-"
        err = bf.get("max_abs_err") or "-"
        lines.append(
            f"| {backend} | {circuit} | {n} | {_gib(c64_mem)} | {_gib(bf_mem)} | {ratio} "
            f"| {c64_s:.2f} | {bf_s:.2f} | {speedup} | {err} |"
        )

    if micro:
        lines += ["", "## Micro (4M bf16 GEMM, single matmul)", ""]
        lines += ["| backend | m | wall s |", "|---|---|---|"]
        for r in sorted(micro, key=lambda x: (x["backend"], int(x["n"]))):
            lines.append(f"| {r['backend']} | {r['n']} | {float(r['wall_s']):.3f} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()
    md = aggregate(args.csv)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
