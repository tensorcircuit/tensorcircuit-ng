"""Phase 0 探针共享骨架。
假设：三探针共用 self-fork + JSON 协议 + 固定宽表 + 计时。
方法：pure 函数，无 GPU 依赖，便于单测。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_common_test.py
"""

from __future__ import annotations
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Callable


def worker_emit(obj: dict) -> None:
    """Worker 打印恰好一行 JSON；orchestrator 解析 lines[-1]。"""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def parse_last_json(stdout: str) -> dict | None:
    """从 stdout 倒序找第一行以 '{' 开头的，json.loads；无则 None。"""
    for ln in reversed(stdout.splitlines()):
        s = ln.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None
    return None


def classify_stderr(stderr: str) -> str:
    """把子进程 stderr 分类为结局标签。"""
    s = (stderr or "").lower()
    if not s:
        return "ok"
    if "out of memory" in s or ("cuda error" in s and "memory" in s):
        return "oom"
    if "overflow" in s or "int32" in s:
        return "crash-int32"
    if "compilation" in s or "xla" in s or "compile" in s:
        return "crash-compile"
    return "crash"


def orchestrate(
    configs, build_worker_argv, script_path, timeout=900, cwd=None, env=None
):
    """每 config fork 一个子进程：[sys.executable, -u, script_path, 'worker', *argv]。
    捕获 stdout，解析末行 JSON。返回 list[dict]，每项 {config, ok, outcome, result|stderr_tail}。"""
    results = []
    for cfg in configs:
        argv = [sys.executable, "-u", script_path, "worker", *build_worker_argv(cfg)]
        try:
            p = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
            )
        except subprocess.TimeoutExpired as e:
            tail = (
                e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            )[-400:]
            results.append(
                {"config": cfg, "ok": False, "outcome": "timeout", "stderr_tail": tail}
            )
            continue
        if p.returncode != 0:
            results.append(
                {
                    "config": cfg,
                    "ok": False,
                    "outcome": classify_stderr(p.stderr),
                    "stderr_tail": (p.stderr or "")[-400:],
                }
            )
            continue
        obj = parse_last_json(p.stdout)
        if obj is None:
            results.append(
                {
                    "config": cfg,
                    "ok": False,
                    "outcome": "crash",
                    "stderr_tail": "[no JSON line] " + (p.stderr or "")[-200:],
                }
            )
            continue
        woutcome = obj.get("outcome")
        if woutcome not in ("run", "ok", None):
            results.append(
                {
                    "config": cfg,
                    "ok": False,
                    "outcome": woutcome,
                    "stderr_tail": "[worker-reported] "
                    + str(obj.get("error", ""))[:300],
                }
            )
            continue
        results.append({"config": cfg, "ok": True, "outcome": "run", "result": obj})
    return results


def fmt_table(headers, rows):
    """固定宽表。"""
    data = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(str(h)), *(len(r[i]) for r in data)) for i, h in enumerate(headers)
    ]
    sep = "  "
    lines = [
        sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
        sep.join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in data:
        lines.append(sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def median_wall_ms(fn, warmup=2, iters=5, sync=None):
    """对 fn 计时，warmup 后取 iters 次的中位数（毫秒）。sync(result) 每次调用后同步。"""
    for _ in range(warmup):
        r = fn()
        if sync is not None:
            sync(r)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn()
        if sync is not None:
            sync(r)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(ts))
