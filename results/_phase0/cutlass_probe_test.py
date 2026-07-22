import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def test_discover_paths_uses_env_vars(monkeypatch):
    import cutlass_probe

    monkeypatch.setenv("CUTLASS_ROOT", "/fake/cutlass")
    monkeypatch.setenv("CUDA_HOME", "/fake/cuda")
    monkeypatch.setenv("NVCC", "/fake/cuda/bin/nvcc")
    p = cutlass_probe.discover_paths()
    assert p["cutlass_root"] == "/fake/cutlass"
    assert p["cuda_home"] == "/fake/cuda"
    assert p["nvcc"] == "/fake/cuda/bin/nvcc"
    assert (
        os.path.isdir(p["cutlass_root"]) is False
    )  # not validated here; build validates


def test_build_extension_signature_exists():
    import cutlass_probe

    assert callable(cutlass_probe.build_extension)


def _gpu_ready():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import cutlass_probe

        p = cutlass_probe.discover_paths()
        return os.path.isdir(p["cutlass_root"]) and os.path.exists(p["nvcc"])
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_build_extension_compiles_and_loads():
    import cutlass_probe

    mod = cutlass_probe.build_extension()
    assert mod.probe() == 42


def test_four_m_coefficients():
    import cutlass_probe

    c = cutlass_probe.four_m_coefficients()
    # ReC = +1*ReA.ReB + (-1)*ImA.ImB ; ImC = +1*ReA.ImB + +1*ImA.ReB
    assert c["rec_rea_reb"] == +1.0 and c["rec_ima_imb"] == -1.0
    assert c["imc_rea_imb"] == +1.0 and c["imc_ima_reb"] == +1.0


def test_c64_reference_matches_numpy_complex():
    import cutlass_probe
    import numpy as np

    rng = np.random.default_rng(0)
    ReA = rng.standard_normal((4, 8)).astype(np.float32)
    ImA = rng.standard_normal((4, 8)).astype(np.float32)
    ReB = rng.standard_normal((8, 6)).astype(np.float32)
    ImB = rng.standard_normal((8, 6)).astype(np.float32)
    ReC, ImC = cutlass_probe.c64_reference(ReA, ImA, ReB, ImB)
    A = (ReA + 1j * ImA).astype(np.complex64)
    B = (ReB + 1j * ImB).astype(np.complex64)
    C = A @ B
    np.testing.assert_allclose(ReC, C.real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ImC, C.imag, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_single_4m_sm80_correctness_real_gemm():
    import cutlass_probe

    r = cutlass_probe.run_single_4m(
        "sm80_fallback", shapes=[(128, 128, 128)], seeds=(0,)
    )
    assert r["correctness"]["gate_pass"] is True, r["correctness"]
    assert r["correctness"]["max_rel"] < 1e-2


def test_sm100_compile_failure_falls_back(monkeypatch):
    import cutlass_probe

    calls = {"n": 0}

    def fake_build(name="cutlass_4m", extra_defines=None):
        calls["n"] += 1
        if extra_defines and "-DCUTLASS_ENABLE_SM100_4M=1" in extra_defines:
            raise RuntimeError("nvcc: sm100 instantiation failed")
        return object()  # non-GPU stub; the sm80 path is mocked below

    # mock _run_sm80 so no real GPU build/run happens — keeps this test GPU-free
    monkeypatch.setattr(cutlass_probe, "build_extension", fake_build)
    monkeypatch.setattr(
        cutlass_probe,
        "_run_sm80",
        lambda shapes, seeds, **kw: {
            "kernel_path": "sm80_fallback",
            "runs": True,
            "correctness": {"gate_pass": True},
            **kw,
        },
    )
    r = cutlass_probe._attempt_sm100_then_sm80(shapes=[(64, 64, 64)], seeds=(0,))
    assert r["kernel_path"] == "sm80_fallback"
    # the recorded blocker must be present so the artifact can explain the fallback
    assert "sm100_blocker" in r and r["sm100_blocker"]
    # the sm100 build attempt must have actually happened (then _run_sm80 is mocked)
    assert calls["n"] == 1


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_sm100_attempt_runs_or_falls_back():
    import cutlass_probe

    r = cutlass_probe.run_single_4m(
        "sm100_native", shapes=[(128, 128, 128)], seeds=(0,)
    )
    assert r["kernel_path"] in ("sm100_native", "sm80_fallback")
    assert r["correctness"]["gate_pass"] is True


def test_sm120_compile_failure_falls_back(monkeypatch):
    """GPU-free: if the native Sm120 build fails (CUTLASS's Sm120 collective is
    F8F6F4-only, so BF16 instantiation refuses to compile), the dispatcher
    records the verbatim blocker and falls back to sm80."""
    import cutlass_probe

    calls = {"n": 0}

    def fake_build(name="cutlass_4m", extra_defines=None):
        calls["n"] += 1
        if extra_defines and "-DCUTLASS_ENABLE_SM120_4M=1" in extra_defines:
            raise RuntimeError(
                "nvcc: static_assert SM120 TmaWarpSpecialized builder "
                "currently only supports F8F6F4 MMA"
            )
        return object()  # non-GPU stub; _run_sm80 is mocked below

    monkeypatch.setattr(cutlass_probe, "build_extension", fake_build)
    monkeypatch.setattr(
        cutlass_probe,
        "_run_sm80",
        lambda shapes, seeds, **kw: {
            "kernel_path": "sm80_fallback",
            "runs": True,
            "correctness": {"gate_pass": True},
            **kw,
        },
    )
    r = cutlass_probe._attempt_sm120_then_sm80(shapes=[(64, 64, 64)], seeds=(0,))
    assert r["kernel_path"] == "sm80_fallback"
    # the recorded blocker must be present so the artifact can explain the fallback
    assert "sm120_blocker" in r and r["sm120_blocker"]
    # the sm120 build attempt must have actually happened (then _run_sm80 is mocked)
    assert calls["n"] == 1


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_sm120_attempt_runs_or_falls_back():
    import cutlass_probe

    r = cutlass_probe.run_single_4m(
        "sm120_native", shapes=[(128, 128, 128)], seeds=(0,)
    )
    assert r["kernel_path"] in ("sm120_native", "sm80_fallback")
    assert r["correctness"]["gate_pass"] is True


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_single_4m_sm80_has_resource_and_latency():
    import cutlass_probe

    r = cutlass_probe.run_single_4m(
        "sm80_fallback", shapes=[(1024, 1024, 1024)], seeds=(0,)
    )
    assert "resource" in r and "latency" in r
    assert (
        r["resource"]["workspace_bytes"] >= 0
    )  # always available (get_workspace_size)
    # registers/occupancy are best-effort via nvcc --res-usage log; None allowed
    assert r["resource"]["registers"] is None or r["resource"]["registers"] > 0
    assert r["latency"]["kernelonly_median_us"] > 0
    assert r["latency"]["c64_baseline_us"] > 0
    assert r["latency"]["ko_ratio_vs_c64"] > 0  # c64_us / 4m_us (fair kernel-only both)


def test_load_grouped_shapes_filters_real_gemm(monkeypatch, tmp_path):
    """GPU-free: load_grouped_shapes picks the real-gemm (min dim>=16), distinct,
    heterogeneous subset from the contraction CSV. Skinny shapes (e.g. 2x2x2) are
    dropped; duplicates collapse; coverage (subset/total) is recorded by
    run_grouped, not here."""
    import cutlass_probe

    csv = tmp_path / "contraction_shapes.csv"
    csv.write_text("M,N,K\n2,2,2\n1024,1024,1024\n16384,1024,1024\n64,64,64\n")
    monkeypatch.setattr(cutlass_probe, "_CONTRACTION_SHAPES_CSV", str(csv))
    shapes = cutlass_probe.load_grouped_shapes()
    ms = [(s["M"], s["N"], s["K"]) for s in shapes]
    assert all(min(m) >= 16 for m in ms)  # real-gemm floor
    assert len(set(ms)) == len(ms)  # distinct (heterogeneous)
    assert (2, 2, 2) not in ms  # skinny dropped


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_run_grouped_returns_valid_status():
    """Grouped GEMM either runs+passes correctness, or returns a clean
    NOT_SUPPORTED/BLOCKED — all three are legitimate verdicts per spec §9."""
    import cutlass_probe

    shapes = cutlass_probe.load_grouped_shapes()
    g = cutlass_probe.run_grouped(shapes)
    assert g["status"] in ("SUPPORTED", "NOT_SUPPORTED", "BLOCKED"), g
    assert "coverage" in g and g["coverage"]["shapes_total"] == len(shapes)
    if g["status"] == "SUPPORTED":
        assert g["correctness"]["gate_pass"] is True
