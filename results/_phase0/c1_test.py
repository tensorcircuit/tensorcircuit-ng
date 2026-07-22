"""Unit tests for C1 judgment + buffer audit (review §5.4, rereview §4, correction Task A).

Run: pytest results/_phase0/c1_test.py -v
"""

from results._phase0.c1 import judge_c1, upsert_csv_row

# Synthetic XLA buffer-assignment dump fragment for the parser unit test (Task A4).
SYNTH_BA = """BufferAssignment:
allocation 5: size 1048576, preallocated-temp:
 value: <100 anchor.1{0} @0> (size=160,offset=0): c64[10,2]
 value: <101 reuse{0} @0> (size=160,offset=0): c64[10,2]
 value: <102 other{0} @0> (size=192,offset=512): s8[192]
anchor.1{0}:10-20
reuse{0}:30-40
other{0}:5-6
"""


def test_parse_buffer_assignment_extracts_alloc_liveness_aliases():
    """Synthetic buffer-assignment text -> allocation_id/size/offset, liveness, aliasing."""
    from results._phase0.c1_buffer_audit import parse_buffer_assignment

    records, by_key, liveness, by_physical = parse_buffer_assignment(SYNTH_BA)
    a = by_key[("anchor.1", "0")]
    assert a["allocation_id"] == 5
    assert a["allocation_size"] == 1048576
    assert a["offset"] == 0
    assert a["value_size"] == 160
    assert a["allocation_kind"] == "preallocated-temp"
    assert liveness[("anchor.1", "0")] == (10, 20)
    # aliasing: anchor.1{0} and reuse{0} share the same physical bytes (alloc 5, offset 0)
    assert len(by_physical[(5, 0)]) == 2
    assert {r["op_name"] for r in by_physical[(5, 0)]} == {"anchor.1", "reuse"}


def test_c1_pass_when_all_conditions_met():
    r = {
        "planned_temp_bytes": 2**24 * 8,
        "runtime_peak_sampled_bytes": 2**24 * 8,
        "full_state_bytes": 2**24 * 8,
    }
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=[r, r, r],
        materialized_buffer_bytes=2**24 * 8,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "PASS", j


def test_c1_fail_when_buffer_below_half_state():
    r = {
        "planned_temp_bytes": 1000,
        "runtime_peak_sampled_bytes": 1000,
        "full_state_bytes": 2**24 * 8,
    }
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=[r, r, r],
        materialized_buffer_bytes=1000,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "FAIL"
    assert "0.5" in j["reason"] or "threshold" in j["reason"].lower()


def test_c1_unknown_when_repeats_unstable():
    # DYNAMIC runtime peak varies across repeats -> unstable (rereview §4.4).
    base = {"planned_temp_bytes": 2**24 * 8, "full_state_bytes": 2**24 * 8}
    unstable = [
        {**base, "runtime_peak_sampled_bytes": 2**24 * 8},
        {**base, "runtime_peak_sampled_bytes": 1000},
        {**base, "runtime_peak_sampled_bytes": 2**24 * 8},
    ]
    j = judge_c1(
        default_result={**base, "runtime_peak_sampled_bytes": 2**24 * 8},
        nofusion_result={**base, "runtime_peak_sampled_bytes": 2**24 * 8},
        repeats_results=unstable,
        materialized_buffer_bytes=2**24 * 8,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "UNKNOWN"


def test_c1_repeat_stability_requires_dynamic_not_static():
    """Three IDENTICAL static planned values must NOT satisfy stability on their own
    (rereview §4.4). With no dynamic sample, condition 6 cannot be established -> UNKNOWN.
    """
    r = {
        "planned_temp_bytes": 2**24 * 8,
        "full_state_bytes": 2**24 * 8,
    }  # no runtime sample
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=[r, r, r],
        materialized_buffer_bytes=2**24 * 8,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "UNKNOWN"
    assert not j["conditions"]["6_repeat_stable"]


def test_parse_tuple_separates_data_and_workspace():
    """Data output vs cuBLAS workspace selected by dtype/index, NOT max bytes: on small
    GEMMs the s8 workspace (192 B) exceeds the c64 data output (160 B)."""
    from results._phase0.c1_buffer_audit import parse_materialized_buffers

    hlo = (
        "%cc = (c64[10,2]{1,0}, s8[192]{0}) custom-call(%a, %b), "
        'custom_call_target="__cublas$gemm"'
    )
    bufs = parse_materialized_buffers(hlo)
    assert len(bufs) == 1, bufs
    b = bufs[0]
    assert b["data_dtype"] == "c64"
    assert b["data_shape"] == [10, 2]
    assert b["data_output_bytes"] == 160
    assert b["workspace_bytes"] == 192
    assert b["data_result_index"] == 0
    assert b["workspace_result_index"] == 1


def test_audit_anchor_has_real_allocation_liveness_and_aliasing():
    """File integration (Task A2/A4): the audit is enriched from the XLA buffer-assignment
    dump with real allocation_id/size/offset, liveness (birth/death), and aliasing.
    """
    from results._phase0.c1_buffer_audit import audit_buffer_assignment

    a = audit_buffer_assignment(24, 10, "default")
    assert a["allocation_source"] == "xla_buffer_assignment", a
    anchor = [b for b in a["buffers"] if b["is_anchor"]]
    assert len(anchor) == 1, a
    anc = anchor[0]
    assert anc["data_dtype"] == "c64"
    assert anc["data_shape"] == [4096, 16384]
    assert anc["data_output_bytes"] == 4096 * 16384 * 8  # 512 MiB
    assert anc["workspace_bytes"] > 0  # distinct from the data output
    # real allocation/liveness from the dump (not fabricated from the SSA name)
    assert anc["allocation_id"] == 11
    assert anc["offset"] == 536956416
    assert anc["birth"] == 1468 and anc["death"] == 1470
    # P (.497) aliases E (.498) at the same physical offset -> temporal reuse
    assert "custom-call.498{0}" in anc["aliases"], anc["aliases"]


def test_measure_case_splits_planned_and_runtime_peak():
    """GPU integration: measure_case must split the static planned temp from a sampled
    runtime peak (rereview §4.2)."""
    from results._phase0.c1 import measure_case

    r = measure_case(24, 10, disable_fusion=False)
    assert "planned_temp_bytes" in r and "runtime_peak_sampled_bytes" in r
    assert r["planned_temp_bytes"] == 1107476216  # the known static figure
    assert r["runtime_peak_sampled_bytes"] > 0


def test_c1_csv_upsert_no_duplicate(tmp_path):
    """upsert_csv_row must UPSERT on the key columns, never append a duplicate case."""
    import csv

    p = str(tmp_path / "x.csv")
    upsert_csv_row(
        p,
        {"n": 24, "depth": 10, "fusion": "default", "peak": 1},
        ["n", "depth", "fusion", "peak"],
    )
    upsert_csv_row(
        p,
        {"n": 24, "depth": 10, "fusion": "default", "peak": 2},
        ["n", "depth", "fusion", "peak"],
    )
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 1 and int(rows[0]["peak"]) == 2  # upsert, not append


def test_upsert_removes_all_duplicate_keys(tmp_path):
    """rerun idempotency: starting from a CSV with MULTIPLE duplicate-key rows, upsert
    must leave exactly one row with the new value (rereview §4.5)."""
    import csv

    p = str(tmp_path / "y.csv")
    # seed two historical duplicate rows + one different key
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "depth", "fusion", "peak"])
        w.writerow([24, 10, "default", 1])
        w.writerow([24, 10, "default", 2])  # duplicate key
        w.writerow([22, 10, "default", 9])  # different key, must survive
    upsert_csv_row(
        p,
        {"n": 24, "depth": 10, "fusion": "default", "peak": 3},
        ["n", "depth", "fusion", "peak"],
        key_cols=["n", "depth", "fusion"],
    )
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2, rows  # the n=22 row + the single new n=24 row
    n24 = [r for r in rows if r["n"] == "24"]
    assert len(n24) == 1 and int(n24[0]["peak"]) == 3


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
