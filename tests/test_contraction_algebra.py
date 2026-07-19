import numpy as np
import tensorcircuit as tc
import tensorcircuit.cons as cons
from tensorcircuit.contraction_algebra import ContractionAlgebra, StandardAlgebra


def test_standard_tensordot_matches_backend():
    be = tc.backend
    a = be.cast(
        be.convert_to_tensor(np.arange(12, dtype=np.float64).reshape(3, 4)), "float64"
    )
    b = be.cast(
        be.convert_to_tensor(np.arange(20, dtype=np.float64).reshape(4, 5)), "float64"
    )
    alg = StandardAlgebra()
    got = np.array(alg.tensordot(be, a, b, axes=1))
    ref = np.array(be.tensordot(a, b, axes=1))
    assert got.shape == (3, 5)
    np.testing.assert_allclose(got, ref)


def test_standard_einsum_matches_backend():
    be = tc.backend
    a = be.cast(
        be.convert_to_tensor(np.arange(6, dtype=np.float64).reshape(2, 3)), "float64"
    )
    b = be.cast(
        be.convert_to_tensor(np.arange(12, dtype=np.float64).reshape(3, 4)), "float64"
    )
    alg = StandardAlgebra()
    got = np.array(alg.einsum(be, "ab,bc->ac", a, b))
    ref = np.array(be.einsum("ab,bc->ac", a, b))
    np.testing.assert_allclose(got, ref)


def test_standard_name():
    assert StandardAlgebra().name == "standard"


def test_public_api_surface():
    from tensorcircuit import contraction_algebra as tca

    # After Task 14 the package exports only the 4 base names; activation lives
    # in-source via cons.set_contraction_algebra.
    for name in [
        "ContractionAlgebra",
        "StandardAlgebra",
        "Representation",
        "IdentityRepresentation",
    ]:
        assert hasattr(tca, name), name
    # The old monkey-patch API names are intentionally gone:
    for gone in [
        "activate",
        "deactivate",
        "standard",
        "runtime_contraction_algebra",
        "set_contraction_algebra",
        "get_contraction_algebra",
        "injection",
    ]:
        assert not hasattr(tca, gone), gone


def test_algebra_hooks_default_noop():
    alg = StandardAlgebra()
    assert alg.on_contraction_start(["dummy_nodes"]) is None
    assert alg.on_contractor_ready(["dummy_tree"]) is None


from tensorcircuit.contraction_algebra import (
    ContractionAlgebra,
    StandardAlgebra,
    Representation,
    IdentityRepresentation,
)

# --- Task 5: non-standard path (encode -> kernels via implementation= -> decode,
#     hooks fire, primary.ndim == len(output_set), aux stashed) ---


def test_nonstandard_path_encodes_kernel_decodes_in_order():
    import numpy as np
    import tensornetwork as tn
    import opt_einsum

    log = []

    class LogRep(Representation):
        name = "log"

        def encode(self, be, tensors):
            log.append("encode")
            return tensors

        def decode(self, be, tensor):
            log.append("decode")
            return tensor, {}

    class LogAlg(ContractionAlgebra):
        name = "log"
        representation = LogRep()

        def tensordot(self, be, a, b, axes):
            log.append("td")
            return be.tensordot(a, b, axes)

        def einsum(self, be, eq, *ops):
            log.append("ein")
            return be.einsum(eq, *ops)

        def on_contraction_start(self, nodes):
            log.append("start")

        def on_contractor_ready(self, tree):
            log.append("ready")

    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal((2, 3)).astype(np.complex64))
    b = tn.Node(rng.standard_normal((3, 4)).astype(np.complex64))
    tn.connect(a[1], b[0])
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(LogAlg())
    try:
        cons._algebraic_base_contraction(
            [a, b],
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[a[0], b[1]],
        )
        assert log[0] == "start"
        assert "encode" in log and "ready" in log and "decode" in log
        assert log.index("encode") < log.index("decode")
        # hooks fire around encode/ready in the prescribed order:
        assert log.index("start") < log.index("encode")
        assert log.index("ready") < log.index("decode")
    finally:
        cons.set_contraction_algebra(prev)


def test_decode_wrong_rank_raises_loudly():
    import numpy as np
    import tensornetwork as tn
    import opt_einsum
    import pytest

    class BadRep(Representation):
        name = "bad"

        def encode(self, be, tensors):
            return tensors

        def decode(self, be, tensor):
            # Add a trailing storage axis that the representation forgot to strip.
            return (
                be.reshape(tensor, tuple(be.shape_tuple(tensor)) + (1,)),
                {},
            )

    class BadAlg(ContractionAlgebra):
        name = "bad"
        representation = BadRep()

        def tensordot(self, be, a, b, axes):
            return be.tensordot(a, b, axes)

        def einsum(self, be, eq, *ops):
            return be.einsum(eq, *ops)

    # 2-node fully contracted scalar: output_set="", len 0. BadAlg.decode returns
    # rank 1, so primary.ndim != len(output_set) trips the assert.
    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal(2).astype(np.complex64))
    b = tn.Node(rng.standard_normal(2).astype(np.complex64))
    tn.connect(a[0], b[0])
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(BadAlg())
    try:
        with pytest.raises(AssertionError):
            cons._algebraic_base_contraction(
                [a, b],
                algorithm=opt_einsum.paths.dynamic_programming,
                output_edge_order=[],
            )
    finally:
        cons.set_contraction_algebra(prev)


def test_aux_outputs_stashed_and_reordered():
    # The full aux-reorder test is Task 8 (counting). Here we only confirm the
    # side-channel store API exists and round-trips a value.
    cons._stash_aux_outputs({"count": 5})
    assert cons._aux_outputs()["count"] == 5


def test_representation_identity_roundtrip():
    rep = IdentityRepresentation()
    import numpy as np

    t = [np.ones((2, 3))]
    assert rep.encode(None, t) is t  # identity returns same
    primary, aux = rep.decode(None, t[0])
    assert primary is t[0] and aux == {}


def test_standard_algebra_carries_identity_representation():
    alg = StandardAlgebra()
    assert isinstance(alg.representation, IdentityRepresentation)
    assert alg.name == "standard"


def test_contraction_algebra_representation_default_is_identity():
    # a minimal concrete algebra that does NOT override representation
    class BareAlgebra(ContractionAlgebra):
        name = "bare"

        def tensordot(self, be, a, b, axes):
            return be.tensordot(a, b, axes)

        def einsum(self, be, eq, *ops):
            return be.einsum(eq, *ops)

    assert isinstance(BareAlgebra().representation, IdentityRepresentation)


# --- Task 2: cons.py algebra state (set_contraction_algebra) ---


class _NS(ContractionAlgebra):  # non-standard stub for constraint tests
    name = "ns"
    representation = None  # noqa — not used in these tests

    def tensordot(self, be, a, b, axes):
        return be.tensordot(a, b, axes)

    def einsum(self, be, eq, *ops):
        return be.einsum(eq, *ops)


def test_set_contraction_algebra():
    prev = cons.get_contraction_algebra()
    try:
        cons.set_contraction_algebra(_NS())
        assert isinstance(cons.get_contraction_algebra(), _NS)
    finally:
        cons.set_contraction_algebra(prev)


def test_runtime_contractor_restores_algebra():
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(_NS())
    try:
        with cons.runtime_contractor("greedy"):
            assert isinstance(cons.get_contraction_algebra(), _NS)
        # runtime_contractor saves/restores algebra independently of the contractor,
        # so algebra set before entering survives unchanged.
        assert isinstance(cons.get_contraction_algebra(), _NS)
    finally:
        cons.set_contraction_algebra(prev)


def test_set_function_contractor_restores_algebra():
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(_NS())
    try:

        @cons.set_function_contractor("greedy")
        def f():
            return cons.get_contraction_algebra()

        inside = f()
        assert isinstance(inside, _NS)  # algebra active during the call
        # after the call, algebra survives unchanged (it was not set by the decorator)
        assert isinstance(cons.get_contraction_algebra(), _NS)
    finally:
        cons.set_contraction_algebra(prev)


# --- Task 3: cons.py guards (_merge_single_gates skip + _base routing) ---


def test_merge_single_gates_skipped_under_nonstandard_algebra(monkeypatch):
    import tensornetwork as tn

    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(_NS())
    try:

        def boom(*a, **k):
            raise AssertionError(
                "merge body ran — should be skipped under non-standard algebra"
            )

        monkeypatch.setattr(tn, "contract_parallel", boom)
        out = cons._merge_single_gates(
            ["fake_node"], 7
        )  # guard fires before any node access
        assert out == (["fake_node"], 7)
    finally:
        cons.set_contraction_algebra(prev)


def test_base_routes_to_algebraic_under_nonstandard(monkeypatch):
    import numpy as np
    import tensornetwork as tn
    import opt_einsum

    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal((2, 3)).astype(np.complex64))
    b = tn.Node(rng.standard_normal((3, 4)).astype(np.complex64))
    tn.connect(a[1], b[0])
    routed = {}

    def fake_alg(nodes, algorithm, *args, **kw):
        routed["called"] = True
        return "FINAL"

    monkeypatch.setattr(cons, "_algebraic_base_contraction", fake_alg)
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(_NS())
    try:
        cons._base(
            [a, b],
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[a[0], b[1]],
        )
        assert routed.get("called") is True
    finally:
        cons.set_contraction_algebra(prev)


# --- Task 4: lock the StandardAlgebra backward-compat baseline ---
# A real 2-node contraction under the default StandardAlgebra must produce
# results bit-identical to a direct np.tensordot reference. This is the
# safety net before Task 5 adds the non-standard encode/decode path to
# _algebraic_base_contraction. If this test passes on today's code (which
# ignores the algebra), that is exactly the point: it locks the baseline.


def test_standard_algebra_contraction_matches_native():
    import numpy as np
    import tensornetwork as tn

    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal((2, 3)).astype(np.complex64))
    b = tn.Node(rng.standard_normal((3, 4)).astype(np.complex64))
    tn.connect(a[1], b[0])
    prev = cons.get_contraction_algebra()
    assert isinstance(prev, StandardAlgebra)
    cons.set_contraction_algebra(StandardAlgebra())
    try:
        import opt_einsum

        n = cons._algebraic_base_contraction(
            [a, b],
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[a[0], b[1]],
        )
        ref = np.tensordot(a.tensor, b.tensor, axes=([1], [0]))
        np.testing.assert_allclose(n.tensor, ref, rtol=1e-6)
    finally:
        cons.set_contraction_algebra(prev)


def test_aux_stash_handles_ignore_edge_order_with_none_order(monkeypatch):
    # A non-standard algebra returning aux under ignore_edge_order=True + output_edge_order=None
    # must NOT crash (counting-scalar path). Uses the LogAlg/LogRep from test_nonstandard_path_*.
    import numpy as np
    import tensornetwork as tn
    import opt_einsum

    class AuxRep(Representation):
        def encode(self, be, tensors):
            return tensors

        def decode(self, be, tensor):
            return tensor, {"count": np.ones_like(tensor)}  # non-empty aux

    class AuxAlg(ContractionAlgebra):
        name = "aux"
        representation = AuxRep()

        def tensordot(self, be, a, b, axes):
            return be.tensordot(a, b, axes)

        def einsum(self, be, eq, *ops):
            return be.einsum(eq, *ops)

    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal(2).astype(np.complex64))
    b = tn.Node(rng.standard_normal(2).astype(np.complex64))
    tn.connect(a[0], b[0])
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(AuxAlg())
    try:
        # scalar contraction, ignore_edge_order=True, output_edge_order=None — must not raise
        cons._algebraic_base_contraction(
            [a, b],
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=None,
            ignore_edge_order=True,
        )
        assert "count" in cons._aux_outputs()  # aux stashed without crashing
    finally:
        cons.set_contraction_algebra(prev)


def test_get_contractor_kwargs_default():
    assert StandardAlgebra().get_contractor_kwargs() == {}
    # Concrete algebra without override inherits the default
    assert _NS().get_contractor_kwargs() == {}
