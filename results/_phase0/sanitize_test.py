"""Unit tests for the unified privacy sanitizer (Task 8, spec §3.7)."""

from __future__ import annotations

import os

import pytest

from results._phase0.sanitize import (
    sanitize_text,
    sanitize_file,
    rehash_c2_checkpoint,
    rehash_numerical_binding,
)

# --- Each substitution ----------------------------------------------------


class TestSanitizeText:
    """Each substitution rule in sanitize_text."""

    def test_home_absolute_path(self):
        """Absolute home dir -> <home>."""
        text = "/home/alice/miniconda3/envs/tcng/bin/nvcc"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "/home/alice" not in out
        assert "<home>" in out

    def test_repo_absolute_path(self):
        """Absolute repo dir -> <repo>."""
        text = "/mnt/e/Study/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu"
        out = sanitize_text(
            text, home="/home/alice", repo="/mnt/e/Study/tensorcircuit-ng"
        )
        assert "/mnt/e/Study/tensorcircuit-ng" not in out
        assert "<repo>" in out
        assert "<repo>/results/_phase0/cpp/cutlass_4m.cu" == out

    def test_dollar_home_placeholder(self):
        """Legacy $HOME placeholder -> <home>."""
        text = "$HOME/miniconda3/envs/tcng/bin/nvcc"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "$HOME" not in out
        assert "<home>" in out

    def test_dollar_repo_placeholder(self):
        """Legacy $REPO placeholder -> <repo>."""
        text = "$REPO/results/_phase0/cpp/cutlass_4m.cu"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "$REPO" not in out
        assert "<repo>" in out

    def test_tilde_slash(self):
        """Shell ~/ shorthand -> <home>/."""
        text = "~/cutlass_spike/include"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "~" not in out
        assert "<home>/<toolchain>/include" == out

    def test_toolchain_dir(self):
        """cutlass_spike -> <toolchain>."""
        text = "$HOME/cutlass_spike/include/cutlass/gemm"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "cutlass_spike" not in out
        assert "<toolchain>" in out

    def test_env_name_tcng(self):
        """tcng -> <env>."""
        text = "envs/tcng/bin/nvcc"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "tcng" not in out
        assert "<env>" in out

    def test_env_name_nvcc_spike(self):
        """nvcc_spike -> <env>."""
        text = "envs/nvcc_spike/bin/nvcc"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "nvcc_spike" not in out
        assert "<env>" in out


# --- Placeholder double-wrap regression (Task 8 review fix) -----------------


class TestPlaceholderDoubleWrapRegression:
    """An already-angle-bracketed private token must NOT double-wrap.

    Regression for the Task 8 review finding: the cutlass recipe string
    ``CUDA_HOME=<nvcc_spike>`` was sanitized to ``CUDA_HOME=<<env>>`` because
    the env-name substitution did a naive ``text.replace("nvcc_spike",
    "<env>")`` on the ``nvcc_spike`` substring *inside* the existing angle
    brackets, producing ``<`` + ``<env>`` + ``>`` = ``<<env>>``.

    The fix replaces the already-bracketed form (``<nvcc_spike>``) before the
    bare form so both ``nvcc_spike`` and ``<nvcc_spike>`` sanitize to exactly
    ``<env>`` (same bracketed-first ordering for ``<cutlass_spike>`` ->
    ``<toolchain>``). This is NOT a blanket ``<<``->``<`` collapse -- it only
    touches the known private tokens, so legitimate C++ template/shift syntax
    (``enable_if_t<<expression>``, ``device_kernel<Sm100GemmKernel>``) is
    preserved.
    """

    def test_env_name_already_bracketed(self):
        """sanitize_text('CUDA_HOME=<nvcc_spike>') -> 'CUDA_HOME=<env>'."""
        out = sanitize_text("CUDA_HOME=<nvcc_spike>", home="/home/alice", repo="/repo")
        assert out == "CUDA_HOME=<env>"
        assert "<<" not in out
        assert ">>" not in out

    def test_env_name_bare_still_works(self):
        """Bare nvcc_spike still sanitizes to <env> (no regression)."""
        out = sanitize_text("CUDA_HOME=nvcc_spike", home="/home/alice", repo="/repo")
        assert out == "CUDA_HOME=<env>"

    def test_tcng_already_bracketed(self):
        """<tcng> (the other env name) does not double-wrap."""
        out = sanitize_text("env=<tcng>", home="/home/alice", repo="/repo")
        assert out == "env=<env>"
        assert "<<" not in out

    def test_toolchain_already_bracketed(self):
        """<cutlass_spike> does not double-wrap into <<toolchain>>."""
        out = sanitize_text(
            "CUTLASS_ROOT=<cutlass_spike>", home="/home/alice", repo="/repo"
        )
        assert out == "CUTLASS_ROOT=<toolchain>"
        assert "<<" not in out

    def test_bracketed_token_does_not_corrupt_cpp_templates(self):
        """Hardening must not touch legitimate C++ template/shift syntax."""
        raw = "device_kernel<Sm100GemmKernel> and std::enable_if_t<<expression>, void>"
        out = sanitize_text(raw, home="/home/alice", repo="/repo")
        # C++ template syntax survives byte-for-byte (no private tokens here).
        assert "device_kernel<Sm100GemmKernel>" in out
        assert "std::enable_if_t<<expression>, void>" in out
        assert "<<" in out  # legitimate, must NOT be collapsed

    def test_recipe_renders_without_double_brackets(self, tmp_path):
        """The cutlass_sm120_4m recipe renders with a clean ``CUDA_HOME=<env>``
        line and no ``<<`` double-bracket anywhere in the recipe (the original
        defect was ``CUDA_HOME=<<env>>``)."""
        from results._phase0.cutlass_probe import write_artifacts

        # Minimal verdict: only the keys write_artifacts reads for the recipe
        # header. The embedded JSON is intentionally blocker-free so the
        # ``no <<`` assertion isolates the recipe line (real blocker text
        # legitimately contains C++ ``enable_if_t<<expression>``).
        verdict = {
            "schema_version": "cutlass-sm120-4m-v1",
            "overall": "FEASIBLE_WITH_SM80_FALLBACK",
        }
        write_artifacts(verdict, str(tmp_path))
        md = (tmp_path / "cutlass_sm120_4m.md").read_text()
        assert "CUDA_HOME=<env>" in md
        assert "CUDA_HOME=<<env>>" not in md
        assert "<<" not in md
        assert ">>" not in md


# --- Preserve-diagnostics guarantee ---------------------------------------


class TestPreserveDiagnostics:
    """The sanitizer MUST preserve diagnostic semantics."""

    def test_cutlass_source_file_refs_preserved(self):
        """CUTLASS source-file references (file:line) survive intact."""
        text = (
            "$HOME/cutlass_spike/include/cutlass/gemm/collective/builders/"
            "sm120_mma_builder.inl(80): error: static assertion failed"
        )
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "sm120_mma_builder.inl(80)" in out
        assert "error: static assertion failed" in out

    def test_mma_sm120_ref_preserved(self):
        """mma_sm120.hpp:47 reference survives."""
        text = "$HOME/cutlass_spike/include/cute/arch/mma_sm120.hpp(47): error"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "mma_sm120.hpp(47)" in out
        assert "error" in out

    def test_f8f6f4_error_text_preserved(self):
        """F8F6F4 collective limit error text survives."""
        text = (
            'static assertion failed with "SM120 TmaWarpSpecialized builder '
            'currently only supports F8F6F4 MMA."'
        )
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "F8F6F4" in out
        assert "SM120 TmaWarpSpecialized builder currently only supports" in out

    def test_cuda_arch_gate_preserved(self):
        """__CUDA_ARCH__==1000 gate text survives."""
        text = "Sm100 device MMA gated by __CUDA_ARCH__==1000"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "__CUDA_ARCH__==1000" in out

    def test_sm100_blocker_preserved(self):
        """kErrorInternal + cudaFuncSetAttribute text survives."""
        text = (
            "Sm100 initialize failed: kErrorInternal -- cudaFuncSetAttribute on "
            "device_kernel<Sm100GemmKernel> fails on sm_120"
        )
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "kErrorInternal" in out
        assert "cudaFuncSetAttribute" in out
        assert "Sm100GemmKernel" in out

    def test_relative_paths_within_repo_preserved(self):
        """Relative paths within the repo (after <repo>) survive."""
        text = "$REPO/results/_phase0/cpp/cutlass_4m.cu"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "<repo>/results/_phase0/cpp/cutlass_4m.cu" == out

    def test_line_numbers_preserved(self):
        """Line numbers in compiler diagnostics survive."""
        text = "$HOME/cutlass_spike/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp(53): warning"
        out = sanitize_text(text, home="/home/alice", repo="/repo")
        assert "sm100_static_tile_scheduler.hpp(53)" in out
        assert "warning" in out

    def test_full_blocker_string_round_trip(self):
        """A realistic blocker string is sanitized without losing diagnostics."""
        raw = (
            "Error building extension 'cutlass_4m_sm120': [1/2] "
            "$HOME/miniconda3/envs/tcng/bin/nvcc -MD -MF cutlass_4m.cuda.o.d "
            "-I$HOME/cutlass_spike/include "
            "-c $REPO/results/_phase0/cpp/cutlass_4m.cu -o cutlass_4m.cuda.o\n"
            "$HOME/cutlass_spike/include/cutlass/gemm/collective/builders/"
            "sm120_mma_builder.inl(80): error: static assertion failed with "
            '"SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA."\n'
            "$HOME/cutlass_spike/include/cute/arch/mma_sm120.hpp(47): error: "
            '"No MMA matches SM120_16x8x32_TN for given data types."\n'
            "3 errors detected in the compilation of "
            '"$REPO/results/_phase0/cpp/cutlass_4m.cu".'
        )
        out = sanitize_text(raw, home="/home/alice", repo="/repo")
        # Private strings gone.
        assert "tcng" not in out
        assert "cutlass_spike" not in out
        assert "nvcc_spike" not in out
        assert "$HOME" not in out
        assert "$REPO" not in out
        assert "/home/alice" not in out
        # Diagnostics preserved.
        assert "sm120_mma_builder.inl(80)" in out
        assert "mma_sm120.hpp(47)" in out
        assert "F8F6F4" in out
        assert "SM120_16x8x32_TN" in out
        assert "3 errors detected" in out
        assert "<repo>/results/_phase0/cpp/cutlass_4m.cu" in out


# --- sanitize_file --------------------------------------------------------


class TestSanitizeFile:
    """sanitize_file in-place sanitization + CRLF normalization."""

    def test_sanitize_file_removes_private_strings(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("$HOME/cutlass_spike/include\n", newline="")
        assert sanitize_file(str(p)) is True
        content = p.read_text()
        assert "$HOME" not in content
        assert "cutlass_spike" not in content
        assert "<home>/<toolchain>/include" in content

    def test_sanitize_file_noop_when_clean(self, tmp_path):
        p = tmp_path / "clean.txt"
        p.write_text("<home>/<toolchain>/include\nno private strings\n", newline="")
        assert sanitize_file(str(p)) is False

    def test_sanitize_file_normalizes_crlf(self, tmp_path):
        p = tmp_path / "crlf.txt"
        p.write_bytes(b"clean line\r\nanother\r\n")
        assert sanitize_file(str(p)) is True
        assert b"\r\n" not in p.read_bytes()
        assert b"\n" in p.read_bytes()

    def test_sanitize_file_preserves_diagnostics(self, tmp_path):
        p = tmp_path / "diag.txt"
        p.write_text(
            "$HOME/cutlass_spike/include/cutlass/gemm/collective/builders/"
            "sm120_mma_builder.inl(80): error: F8F6F4\n",
            newline="",
        )
        sanitize_file(str(p))
        content = p.read_text()
        assert "sm120_mma_builder.inl(80)" in content
        assert "F8F6F4" in content


# --- rehash_c2_checkpoint -------------------------------------------------


class TestRehashC2Checkpoint:
    """rehash_c2_checkpoint updates hashes after sanitization."""

    def test_rehash_updates_source_hlo_hash(self, tmp_path):
        """After sanitizing the HLO file, the checkpoint hash is updated."""
        import hashlib
        import json

        base = str(tmp_path)
        # Create a sanitized HLO file.
        hlo_content = "HLO with <repo>/tensorcircuit/backends/jax_backend.py\n"
        hlo_path = os.path.join(base, "c1_optimized_hlo", "n24_d10_exp_default.hlo")
        os.makedirs(os.path.dirname(hlo_path))
        with open(hlo_path, "w") as fh:
            fh.write(hlo_content)

        # Create a buffer-assignment file.
        ba_content = "buffer-assignment with <repo>/tensorcircuit\n"
        ba_path = os.path.join(base, "c1_xla_dump", "n24_d10_default", "ba.txt")
        os.makedirs(os.path.dirname(ba_path))
        with open(ba_path, "w") as fh:
            fh.write(ba_content)

        # Create c2_judgment.json with artifact_paths.
        c2j = {
            "n24_d10_default": {
                "artifact_paths": {
                    "source_hlo": "results/phase0/c1_optimized_hlo/n24_d10_exp_default.hlo",
                    "buffer_assignment": "results/phase0/c1_xla_dump/n24_d10_default/ba.txt",
                }
            }
        }
        with open(os.path.join(base, "c2_judgment.json"), "w") as fh:
            json.dump(c2j, fh)

        # Create c2_checkpoint_manifest.json with stale hashes.
        ckpt = {
            "artifact_hashes": {
                "source_hlo": "stale_hash_0000",
                "buffer_assignment": "stale_hash_0001",
            }
        }
        with open(os.path.join(base, "c2_checkpoint_manifest.json"), "w") as fh:
            json.dump(ckpt, fh)

        # Rehash.
        assert rehash_c2_checkpoint(base) is True

        # Verify hashes updated.
        with open(os.path.join(base, "c2_checkpoint_manifest.json")) as fh:
            updated = json.load(fh)
        expected_hlo = hashlib.sha256(hlo_content.encode()).hexdigest()
        expected_ba = hashlib.sha256(ba_content.encode()).hexdigest()
        assert updated["artifact_hashes"]["source_hlo"] == expected_hlo
        assert updated["artifact_hashes"]["buffer_assignment"] == expected_ba

    def test_rehash_noop_when_hashes_match(self, tmp_path):
        """When ALL hashes already match, rehash returns False."""
        import hashlib
        import json

        base = str(tmp_path)

        # Create all C2 checkpoint source files with correct hashes.
        files = {
            "c1_optimized_hlo/n24_d10_exp_default.hlo": "clean HLO\n",
            "c1_xla_dump/n24_d10_default/ba.txt": "clean BA\n",
            "c1_buffer_assignment/n24_d10_default.json": "clean audit\n",
            "c1_c2_edge_map.json": "clean edge\n",
            "c2_peak_frontier.json": "clean frontier\n",
            "region_prototype.json": "clean proto\n",
            "c2_judgment.json": '{"n24_d10_default": {}}',
        }
        for rel, content in files.items():
            full = os.path.join(base, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(content)

        c2j = {
            "n24_d10_default": {
                "artifact_paths": {
                    "source_hlo": "results/phase0/c1_optimized_hlo/n24_d10_exp_default.hlo",
                    "buffer_assignment": "results/phase0/c1_xla_dump/n24_d10_default/ba.txt",
                    "audit": "results/phase0/c1_buffer_assignment/n24_d10_default.json",
                    "edge_map": "results/phase0/c1_c2_edge_map.json",
                    "peak_frontier": "results/phase0/c2_peak_frontier.json",
                    "prototype": "results/phase0/region_prototype.json",
                }
            }
        }
        with open(os.path.join(base, "c2_judgment.json"), "w") as fh:
            json.dump(c2j, fh)

        # Compute correct hashes for all keys.
        correct_hashes = {}
        for key, rel in [
            ("source_hlo", "c1_optimized_hlo/n24_d10_exp_default.hlo"),
            ("buffer_assignment", "c1_xla_dump/n24_d10_default/ba.txt"),
            ("allocation_audit", "c1_buffer_assignment/n24_d10_default.json"),
            ("edge_map", "c1_c2_edge_map.json"),
            ("peak_frontier", "c2_peak_frontier.json"),
            ("prototype", "region_prototype.json"),
            ("c2_judgment", "c2_judgment.json"),
        ]:
            with open(os.path.join(base, rel), "rb") as fh:
                correct_hashes[key] = hashlib.sha256(fh.read()).hexdigest()

        ckpt = {"artifact_hashes": correct_hashes}
        with open(os.path.join(base, "c2_checkpoint_manifest.json"), "w") as fh:
            json.dump(ckpt, fh)

        assert rehash_c2_checkpoint(base) is False


# --- rehash_numerical_binding ---------------------------------------------


class TestRehashNumericalBinding:
    """rehash_numerical_binding updates case_binding hashes after sanitization.

    Task 5: the binding now covers ALL 9 route-source files (full sha256, new
    ``_sha256`` key names matching ``manifest.NUMERICAL_BINDINGS``). These
    tests create all 9 source files and verify the rehash handles the full set.
    """

    # (filename, content) for all 9 NUMERICAL_BINDINGS source files.
    _FILES = [
        ("c1_c2_edge_map.json", '{"edge": "sanitized <repo>"}'),
        ("region_prototype.json", '{"proto": 1}'),
        ("contraction_shapes.csv", "M,N,K\n16,16,16\n"),
        ("cublaslt_planar_capability.json", '{"planar": 1}'),
        ("cublaslt_full_matrix.csv", "M,N,K,status\n16,16,16,ok\n"),
        ("cublaslt_grouped_capability.json", '{"grouped": 1}'),
        ("cublaslt_grouped.csv", "route,M,N,K\nplanar,16,16,16\n"),
        ("cutlass_sm120_4m.json", '{"single_4m": 1}'),
        ("numerical_validation.csv", "route,relative_l2\nplanar,1e-5\n"),
    ]

    def _write_files(self, base):
        for rel, content in self._FILES:
            with open(os.path.join(base, rel), "w") as fh:
                fh.write(content)

    def _correct_hashes(self):
        import hashlib

        return {
            "edge_map_sha256": hashlib.sha256(self._FILES[0][1].encode()).hexdigest(),
            "region_prototype_sha256": hashlib.sha256(
                self._FILES[1][1].encode()
            ).hexdigest(),
            "contraction_shapes_sha256": hashlib.sha256(
                self._FILES[2][1].encode()
            ).hexdigest(),
            "cublaslt_planar_capability_sha256": hashlib.sha256(
                self._FILES[3][1].encode()
            ).hexdigest(),
            "cublaslt_full_matrix_sha256": hashlib.sha256(
                self._FILES[4][1].encode()
            ).hexdigest(),
            "cublaslt_grouped_capability_sha256": hashlib.sha256(
                self._FILES[5][1].encode()
            ).hexdigest(),
            "cublaslt_grouped_rows_sha256": hashlib.sha256(
                self._FILES[6][1].encode()
            ).hexdigest(),
            "cutlass_4m_sha256": hashlib.sha256(self._FILES[7][1].encode()).hexdigest(),
            "numerical_csv_sha256": hashlib.sha256(
                self._FILES[8][1].encode()
            ).hexdigest(),
        }

    def test_rehash_updates_edge_map_hash(self, tmp_path):
        """After c1_c2_edge_map.json is regenerated, the binding hash is updated."""
        import hashlib
        import json

        base = str(tmp_path)
        self._write_files(base)
        correct = self._correct_hashes()
        # Stale edge_map_sha256; all other 8 correct.
        stale_binding = {"algorithm": "sha256", **correct}
        stale_binding["edge_map_sha256"] = "0" * 64
        nv = {"case_binding": stale_binding}
        with open(os.path.join(base, "numerical_validation.json"), "w") as fh:
            json.dump(nv, fh)

        assert rehash_numerical_binding(base) is True

        with open(os.path.join(base, "numerical_validation.json")) as fh:
            updated = json.load(fh)
        assert updated["case_binding"]["edge_map_sha256"] == correct["edge_map_sha256"]

    def test_rehash_noop_when_hashes_match(self, tmp_path):
        """When all 9 case_binding hashes match, rehash returns False."""
        import json

        base = str(tmp_path)
        self._write_files(base)
        nv = {"case_binding": {"algorithm": "sha256", **self._correct_hashes()}}
        with open(os.path.join(base, "numerical_validation.json"), "w") as fh:
            json.dump(nv, fh)

        assert rehash_numerical_binding(base) is False


# --- LF pin regression (Task 8: kill OneDrive CRLF phantoms) ---------------


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# The two CSVs that historically drifted to content-less M (CRLF) after every
# non-GPU suite run (brief: "this has failed twice before, be rigorous").
_PHANTOM_CSVS = (
    "results/phase0/c1_c2_edge_map.csv",
    "results/phase0/c2_peak_windows.csv",
)


class TestLFPinRegression:
    """Lock in the LF pin so the OneDrive CRLF phantom stays dead.

    Root fix (brief deliverable 5): ``.gitattributes`` pins ``eol=lf`` for
    ``results/phase0/**/*.{csv,json,hlo,txt,md}`` (one line per extension --
    gitattributes has no ``{a,b}`` brace expansion) AND every CSV generator
    writes with ``lineterminator="\\n"`` / ``newline="\\n"`` so the working-copy
    bytes are LF even before git normalizes.  These tests guard both halves.
    """

    def test_gitattributes_pins_lf_for_phantom_csvs(self):
        """``git check-attr eol`` reports ``lf`` for both phantom CSVs."""
        import subprocess

        for rel in _PHANTOM_CSVS:
            out = subprocess.check_output(
                ["git", "check-attr", "eol", rel],
                cwd=_REPO_ROOT,
                text=True,
            )
            assert (
                "eol: lf" in out
            ), f"gitattributes does not pin eol=lf for {rel}:\n{out}"

    def test_phantom_csvs_have_no_crlf_bytes(self):
        """On-disk phantom CSVs carry no CRLF (generator writes LF)."""
        for rel in _PHANTOM_CSVS:
            full = os.path.join(_REPO_ROOT, *rel.split("/"))
            with open(full, "rb") as fh:
                content = fh.read()
            assert (
                b"\r\n" not in content
            ), f"{rel} contains CRLF bytes (OneDrive phantom regressed)"


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.9: sanitizer source hardcodes private names.
# The sanitizer must extract env/toolchain names dynamically from
# CONDA_PREFIX / CUDA_HOME / CUTLASS_ROOT / home / repo, not hardcode them.
# ---------------------------------------------------------------------------


def test_sanitize_defaults_do_not_hardcode_private_names():
    """Nongpu rereview finding 3.9: the sanitizer must NOT hardcode real env /
    toolchain names as default module constants. It must extract them
    dynamically from ``CONDA_PREFIX`` / ``CUDA_HOME`` / ``CUTLASS_ROOT`` /
    home / repo. Current source hardcodes ``_ENV_NAMES`` and
    ``_TOOLCHAIN_DIRS`` with real private names (sanitize.py:40,43)."""
    from results._phase0 import sanitize

    env_names = getattr(sanitize, "_ENV_NAMES", ())
    toolchain_dirs = getattr(sanitize, "_TOOLCHAIN_DIRS", ())
    # The defaults must be empty -- the sanitizer must extract names dynamically,
    # not hardcode them as module-level constants.
    assert (
        len(env_names) == 0
    ), f"_ENV_NAMES must be empty (dynamic extraction), got {env_names!r}"
    assert (
        len(toolchain_dirs) == 0
    ), f"_TOOLCHAIN_DIRS must be empty (dynamic extraction), got {toolchain_dirs!r}"


def test_probe_sources_do_not_hardcode_private_names_from_sanitizer():
    """Nongpu rereview finding 3.9: ``cutlass_probe.py`` and
    ``cpp/cutlass_4m.cu`` must NOT hardcode real env/toolchain names. The scan
    patterns are read dynamically from the sanitize module's constants (while
    they exist); the fix removes the constants so the scan becomes a no-op.

    This test does NOT hardcode any real names -- it reads them from the
    sanitize module (which currently hardcodes them) and checks the probe
    sources. If the fix removes the constants, the test passes trivially."""
    from results._phase0 import sanitize

    # Read the private names from the sanitize module's constants (if they
    # exist). The fix removes these constants; while they exist, the probe
    # sources must not contain them.
    env_names = getattr(sanitize, "_ENV_NAMES", ())
    toolchain_dirs = getattr(sanitize, "_TOOLCHAIN_DIRS", ())
    private_names = tuple(env_names) + tuple(toolchain_dirs)
    if not private_names:
        return  # fix applied: no hardcoded names to scan for

    tracked_sources = [
        "results/_phase0/cutlass_probe.py",
        "results/_phase0/cpp/cutlass_4m.cu",
    ]
    violations = []
    for rel in tracked_sources:
        full = os.path.join(_REPO_ROOT, *rel.split("/"))
        with open(full, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        for name in private_names:
            if name in content:
                violations.append((rel, name))
    assert not violations, (
        "tracked probe source hardcodes private names (must use dynamic "
        "extraction): " + ", ".join(f"{rel}:{name}" for rel, name in violations)
    )


def test_sanitize_text_supports_fictional_dynamic_names():
    """Nongpu rereview finding 3.9 (complementary GREEN pin): the sanitizer
    must support dynamic env/toolchain names (not just hardcoded ones). Verified
    with FICTIONAL names per the brief (``example-env-alpha``,
    ``example-toolchain-beta``). This already passes on current code
    (``sanitize_text`` accepts ``env_names`` / ``toolchain_dirs`` parameters)."""
    out = sanitize_text(
        "/home/user/envs/example-env-alpha/bin/tool "
        "-I/home/user/example-toolchain-beta/include",
        home="/home/user",
        repo="/repo",
        env_names=("example-env-alpha",),
        toolchain_dirs=("example-toolchain-beta",),
    )
    assert "example-env-alpha" not in out
    assert "example-toolchain-beta" not in out
    assert "<env>" in out
    assert "<toolchain>" in out
