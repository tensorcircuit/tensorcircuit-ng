"""
Tests for ``_pure_tree_flatten``, ``_pure_tree_unflatten``, and ``_pure_tree_map``
in ``tensorcircuit.backends.abstract_backend``.

All tests compare our pure-Python pytree utilities against JAX ``tree_util``
semantics, covering:
  - dict key ordering (sorted vs insertion order)
  - ``None`` as empty pytree (0 leaves)
  - ``namedtuple`` vs plain ``tuple`` type distinction
  - strict structure matching in ``tree_map`` (no scalar broadcast)
  - leaf-count validation in ``unflatten`` (ValueError on mismatch)
  - round-trip ``flatten → unflatten``
  - complex nested structures
"""

import sys
import os
from collections import namedtuple

import numpy as np
import pytest

from jax import tree_util

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))
sys.path.insert(0, modulepath)

import tensorcircuit as tc
from tensorcircuit.backends.abstract_backend import (
    _pure_tree_flatten,
    _pure_tree_map,
    _pure_tree_unflatten,
    _LEAF,
    _NONE_TYPE,
)

Point = namedtuple("Point", ["x", "y"])


# ---------------------------------------------------------------------------
# _pure_tree_flatten
# ---------------------------------------------------------------------------


class TestFlattenBasic:
    """Basic flattening behaviours."""

    def test_flatten_single_int(self):
        leaves, td = _pure_tree_flatten(42)
        assert leaves == [42]
        assert td.typ is _LEAF

    def test_flatten_single_string(self):
        # strings are leaves (not treated as containers)
        leaves, _ = _pure_tree_flatten("hello")
        assert leaves == ["hello"]

    def test_flatten_empty_list(self):
        leaves, td = _pure_tree_flatten([])
        assert leaves == []
        assert td.num_leaves == 0

    def test_flatten_empty_tuple(self):
        leaves, td = _pure_tree_flatten(())
        assert leaves == []
        assert td.num_leaves == 0

    def test_flatten_empty_dict(self):
        leaves, td = _pure_tree_flatten({})
        assert leaves == []
        assert td.num_leaves == 0

    def test_flatten_flat_list(self):
        leaves, td = _pure_tree_flatten([1, 2, 3])
        assert leaves == [1, 2, 3]
        assert td.num_leaves == 3

    def test_flatten_flat_tuple(self):
        leaves, td = _pure_tree_flatten((10, 20))
        assert leaves == [10, 20]
        assert td.num_leaves == 2

    def test_flatten_flat_dict(self):
        leaves, td = _pure_tree_flatten({"x": 1, "y": 2})
        # keys should be sorted → [1, 2]
        assert leaves == [1, 2]
        assert td.num_leaves == 2
        assert td.sorted_keys == ("x", "y")


class TestFlattenNone:
    """``None`` is an empty pytree (0 leaves), not a leaf."""

    def test_none_is_empty(self):
        leaves, td = _pure_tree_flatten(None)
        assert leaves == []
        assert td.typ is _NONE_TYPE
        assert td.num_leaves == 0

    def test_nested_none_in_list(self):
        leaves, _ = _pure_tree_flatten([1, None, 3])
        assert leaves == [1, 3]

    def test_nested_none_in_dict(self):
        leaves, _ = _pure_tree_flatten({"a": 1, "b": None})
        assert leaves == [1]


class TestFlattenDictKeyOrder:
    """Dict keys are iterated in sorted order for deterministic leaf order."""

    def test_unsorted_keys(self):
        tree = {"b": 2, "a": 1}
        leaves, td = _pure_tree_flatten(tree)
        # JAX sorts dict keys; leaves should follow sorted order
        assert leaves == [1, 2]
        assert td.sorted_keys == ("a", "b")

    def test_numeric_string_keys(self):
        tree = {"2": "x", "1": "y", "3": "z"}
        leaves, _ = _pure_tree_flatten(tree)
        assert leaves == ["y", "x", "z"]


class TestFlattenNamedTuple:
    """namedtuple should be a separate type from plain tuple."""

    def test_namedtuple_preserves_type(self):
        p = Point(1, 2)
        leaves, td = _pure_tree_flatten(p)
        assert leaves == [1, 2]
        assert td.typ is Point
        assert td.num_leaves == 2

    def test_namedtuple_is_not_plain_tuple(self):
        p = Point(1, 2)
        _, td_p = _pure_tree_flatten(p)
        _, td_t = _pure_tree_flatten((1, 2))
        assert td_p.typ is Point
        assert td_t.typ is tuple
        assert td_p.typ is not td_t.typ


class TestFlattenNested:
    """Complex nested structures."""

    def test_nested_list_dict(self):
        tree = {"a": [1, 2], "b": (3, {"c": 4})}
        leaves, td = _pure_tree_flatten(tree)
        # sorted keys: a → [1,2], b → (3, {"c": 4})
        assert leaves == [1, 2, 3, 4]
        assert td.num_leaves == 4

    def test_deep_nesting(self):
        tree = {"z": {"y": {"x": [1]}}}
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [1]
        assert td.num_leaves == 1

    def test_mixed_none_and_values(self):
        tree = {"a": None, "b": [None, 1], "c": 3}
        leaves, _ = _pure_tree_flatten(tree)
        # sorted keys: a→None(0 leaves), b→[None, 1] → [1], c→3
        assert leaves == [1, 3]


# ---------------------------------------------------------------------------
# _pure_tree_unflatten
# ---------------------------------------------------------------------------


class TestUnflattenBasic:
    """Basic unflattening behaviours."""

    def test_roundtrip_int(self):
        leaves, td = _pure_tree_flatten(42)
        result = _pure_tree_unflatten(td, leaves)
        assert result == 42

    def test_roundtrip_list(self):
        tree = [1, 2, 3]
        leaves, td = _pure_tree_flatten(tree)
        assert _pure_tree_unflatten(td, leaves) == [1, 2, 3]

    def test_roundtrip_tuple(self):
        tree = (10, 20)
        leaves, td = _pure_tree_flatten(tree)
        assert _pure_tree_unflatten(td, leaves) == (10, 20)

    def test_roundtrip_dict(self):
        tree = {"b": 2, "a": 1}
        leaves, td = _pure_tree_flatten(tree)
        result = _pure_tree_unflatten(td, leaves)
        assert result == {"b": 2, "a": 1}

    def test_roundtrip_empty(self):
        for tree in [[], (), {}]:
            leaves, td = _pure_tree_flatten(tree)
            result = _pure_tree_unflatten(td, leaves)
            assert result == tree

    def test_roundtrip_none(self):
        leaves, td = _pure_tree_flatten(None)
        result = _pure_tree_unflatten(td, leaves)
        assert result is None

    def test_does_not_mutate_input(self):
        tree = {"a": 1, "b": [2, 3]}
        leaves, td = _pure_tree_flatten(tree)
        original_leaves = list(leaves)
        _pure_tree_unflatten(td, leaves)
        assert leaves == original_leaves


class TestUnflattenNamedTuple:
    """namedtuple round-trips correctly."""

    def test_roundtrip_namedtuple(self):
        p = Point(10, 20)
        leaves, td = _pure_tree_flatten(p)
        result = _pure_tree_unflatten(td, leaves)
        assert isinstance(result, Point)
        assert result == Point(10, 20)

    def test_namedtuple_not_plain_tuple(self):
        p = Point(1, 2)
        _, td = _pure_tree_flatten(p)
        result = _pure_tree_unflatten(td, [100, 200])
        assert isinstance(result, Point)
        assert result == Point(100, 200)


class TestUnflattenNested:
    """Nested round-trips."""

    def test_nested_list_dict(self):
        tree = {"a": [1, 2], "b": (3, {"c": 4})}
        leaves, td = _pure_tree_flatten(tree)
        result = _pure_tree_unflatten(td, leaves)
        assert result == tree

    def test_replace_values(self):
        tree = {"a": [1, 2], "b": [3]}
        leaves, td = _pure_tree_flatten(tree)
        # Replace with doubled values
        result = _pure_tree_unflatten(td, [x * 2 for x in leaves])
        assert result == {"a": [2, 4], "b": [6]}


class TestUnflattenLeafCountValidation:
    """Validate leaf count on unflatten (JAX raises ValueError)."""

    def test_extra_leaves_raises_valueerror(self):
        _, td = _pure_tree_flatten({"a": 1})
        with pytest.raises(ValueError, match="Too many leaves"):
            _pure_tree_unflatten(td, [10, 11])

    def test_too_few_leaves_raises_valueerror(self):
        _, td = _pure_tree_flatten({"a": 1, "b": 2})
        with pytest.raises(ValueError):
            _pure_tree_unflatten(td, [10])

    def test_empty_leaves_for_nonempty_tree(self):
        _, td = _pure_tree_flatten([1, 2])
        with pytest.raises(ValueError):
            _pure_tree_unflatten(td, [])

    def test_correct_count_succeeds(self):
        _, td = _pure_tree_flatten({"a": 1, "b": 2})
        result = _pure_tree_unflatten(td, [10, 20])
        assert result == {"a": 10, "b": 20}


# ---------------------------------------------------------------------------
# _pure_tree_map
# ---------------------------------------------------------------------------


class TestTreeMapBasic:
    """Basic mapping behaviours."""

    def test_map_over_scalar(self):
        result = _pure_tree_map(lambda x: x + 1, 5)
        assert result == 6

    def test_map_over_list(self):
        result = _pure_tree_map(lambda x: x * 2, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_map_over_tuple(self):
        result = _pure_tree_map(lambda x: x + 10, (1, 2))
        assert result == (11, 12)

    def test_map_over_dict(self):
        result = _pure_tree_map(lambda x: x * 3, {"a": 1, "b": 2})
        assert result == {"a": 3, "b": 6}

    def test_map_over_nested(self):
        tree = {"a": [1, 2], "b": (3,)}
        result = _pure_tree_map(lambda x: x + 10, tree)
        assert result == {"a": [11, 12], "b": (13,)}


class TestTreeMapMultiArg:
    """Mapping with multiple tree arguments."""

    def test_two_lists(self):
        result = _pure_tree_map(lambda x, y: x + y, [1, 2], [10, 20])
        assert result == [11, 22]

    def test_two_dicts(self):
        result = _pure_tree_map(
            lambda x, y: x + y, {"a": 1, "b": 2}, {"a": 10, "b": 20}
        )
        assert result == {"a": 11, "b": 22}

    def test_two_nested(self):
        t1 = {"a": [1, 2], "b": [3]}
        t2 = {"a": [10, 20], "b": [30]}
        result = _pure_tree_map(lambda x, y: x + y, t1, t2)
        assert result == {"a": [11, 22], "b": [33]}

    def test_three_args(self):
        result = _pure_tree_map(lambda x, y, z: x + y + z, [1], [10], [100])
        assert result == [111]


class TestTreeMapStructureMismatch:
    """JAX raises ValueError on structure mismatch."""

    def test_scalar_broadcast_is_rejected(self):
        """Passing a scalar where a container is expected should fail."""
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), {"a": [1, 2]}, 9)

    def test_none_vs_scalar_is_rejected(self):
        """``None`` is an empty pytree; mapping it with a scalar leaf mismatches."""
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), None, 5)

    def test_namedtuple_vs_tuple_is_rejected(self):
        """namedtuple and plain tuple are different types."""
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), Point(1, 2), (3, 4))

    def test_dict_key_mismatch(self):
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), {"a": 1}, {"b": 2})

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), [1, 2], [1, 2, 3])

    def test_type_mismatch_dict_vs_list(self):
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: (x, y), {"a": 1}, [1])


class TestTreeMapNoArgs:
    """Edge case: no pytree arguments."""

    def test_zero_args_raises(self):
        with pytest.raises(TypeError):
            _pure_tree_map(lambda: 42)


# ---------------------------------------------------------------------------
# Integration / round-trip with backend API
# ---------------------------------------------------------------------------


class TestBackendIntegration:
    """Smoke tests through the ``tc.backend`` public API."""

    def test_flatten_dict_key_order_via_backend(self, npb):
        tree = {"b": 2, "a": 1}
        leaves, _ = tc.backend.tree_flatten(tree)
        assert leaves == [1, 2]

    def test_roundtrip_via_backend(self, npb):
        tree = {"a": [1, 2], "b": (3, 4)}
        leaves, td = tc.backend.tree_flatten(tree)
        result = tc.backend.tree_unflatten(td, leaves)
        assert result == tree

    def test_tree_map_via_backend(self, npb):
        tree = {"a": np.ones([2]), "b": np.zeros([3])}
        result = tc.backend.tree_map(lambda x: x + 1, tree)
        np.testing.assert_allclose(result["a"], 2 * np.ones([2]))
        np.testing.assert_allclose(result["b"], np.ones([3]))


# ---------------------------------------------------------------------------
# Additional corner cases
# ---------------------------------------------------------------------------


class TestFlattenCornerCases:
    """Edge-case leaf and container types."""

    def test_float_is_leaf(self):
        leaves, td = _pure_tree_flatten(3.14)
        assert leaves == [3.14]
        assert td.typ is _LEAF

    def test_bool_is_leaf(self):
        leaves, _ = _pure_tree_flatten(True)
        assert leaves == [True]

    def test_object_is_leaf(self):
        obj = object()
        leaves, _ = _pure_tree_flatten(obj)
        assert leaves == [obj]

    def test_single_element_tuple(self):
        leaves, td = _pure_tree_flatten((42,))
        assert leaves == [42]
        result = _pure_tree_unflatten(td, leaves)
        assert result == (42,)
        assert isinstance(result, tuple)

    def test_single_element_list(self):
        leaves, td = _pure_tree_flatten([99])
        assert leaves == [99]
        result = _pure_tree_unflatten(td, leaves)
        assert result == [99]

    def test_list_of_none(self):
        leaves, td = _pure_tree_flatten([None, None])
        assert leaves == []
        assert td.num_leaves == 0
        result = _pure_tree_unflatten(td, [])
        assert result == [None, None]

    def test_namedtuple_with_none_fields(self):
        p = Point(None, 1)
        leaves, td = _pure_tree_flatten(p)
        assert leaves == [1]
        result = _pure_tree_unflatten(td, leaves)
        assert result == Point(None, 1)

    def test_dict_with_empty_values(self):
        tree = {"a": [], "b": [1]}
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [1]
        result = _pure_tree_unflatten(td, leaves)
        assert result == {"a": [], "b": [1]}

    def test_deeply_nested_list(self):
        tree = [[[[[42]]]]]
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [42]
        result = _pure_tree_unflatten(td, leaves)
        assert result == [[[[[42]]]]]

    def test_mixed_types_in_list(self):
        """Container types become leaves when inside a list."""
        tree = [1, "hello", 3.14, True]
        leaves, _ = _pure_tree_flatten(tree)
        assert leaves == [1, "hello", 3.14, True]

    def test_dict_with_integer_keys(self):
        """Integer dict keys should be sorted numerically."""
        tree = {2: "b", 1: "a", 3: "c"}
        leaves, _ = _pure_tree_flatten(tree)
        assert leaves == ["a", "b", "c"]

    def test_frozenset_is_leaf(self):
        """frozenset is not a supported container; it should be a leaf."""
        fs = frozenset([1, 2, 3])
        leaves, td = _pure_tree_flatten(fs)
        assert leaves == [fs]
        assert td.typ is _LEAF

    def test_dict_subclass_is_container(self):
        """Custom dict subclass is treated as a container (via isinstance)."""

        class MyDict(dict):
            pass

        tree = MyDict({"a": 1})
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [1]
        assert td.num_leaves == 1

    def test_list_subclass_is_container(self):
        """Custom list subclass is treated as a container (via isinstance)."""

        class MyList(list):
            pass

        tree = MyList([1, 2])
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [1, 2]
        assert td.num_leaves == 2


class TestUnflattenCornerCases:
    """Corner cases for unflatten."""

    def test_treedef_is_reusable(self):
        """The same _TreeDef should be usable multiple times."""
        _, td = _pure_tree_flatten({"a": 1, "b": 2})
        r1 = _pure_tree_unflatten(td, [10, 20])
        r2 = _pure_tree_unflatten(td, [30, 40])
        assert r1 == {"a": 10, "b": 20}
        assert r2 == {"a": 30, "b": 40}

    @pytest.mark.parametrize(
        "tree", [None, [], {}], ids=["none", "empty_list", "empty_dict"]
    )
    def test_unflatten_empty_with_extra_leaves_raises(self, tree):
        """0-leaf structures should reject any non-empty leaf list."""
        _, td = _pure_tree_flatten(tree)
        with pytest.raises(ValueError):
            _pure_tree_unflatten(td, [1])

    def test_complex_nested_roundtrip_with_none(self):
        tree = {"a": [None, {"b": 1}], "c": (None, [2])}
        leaves, td = _pure_tree_flatten(tree)
        assert leaves == [1, 2]
        result = _pure_tree_unflatten(td, leaves)
        assert result == tree

    def test_unflatten_dict_key_to_sorted_leaf_mapping(self):
        """unflatten maps leaves in sorted-key order back to dict keys."""
        tree = {"z": 1, "a": 2, "m": 3}
        leaves, td = _pure_tree_flatten(tree)
        # leaves are in sorted-key order: a→2, m→3, z→1
        assert leaves == [2, 3, 1]
        # unflatten also consumes leaves in sorted-key order
        result = _pure_tree_unflatten(td, [20, 30, 40])
        assert result == {"z": 40, "a": 20, "m": 30}


class TestTreeMapCornerCases:
    """Corner cases for tree_map."""

    def test_map_two_none_arguments(self):
        result = _pure_tree_map(lambda x, y: (x, y), None, None)
        assert result is None

    def test_map_two_all_scalar_arguments(self):
        result = _pure_tree_map(lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_nested_structure_mismatch_deep(self):
        """Mismatch at a deeply nested level should be detected."""
        t1 = {"a": {"b": [1, 2]}}
        t2 = {"a": {"b": [1, 2, 3]}}  # deeper mismatch
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: x + y, t1, t2)

    def test_nested_type_mismatch(self):
        """Type mismatch at a nested level."""
        t1 = {"a": [1, 2]}
        t2 = {"a": (1, 2)}  # list vs tuple at nested level
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: x + y, t1, t2)

    def test_different_namedtuple_types_mismatch(self):
        """Two different namedtuple types should not match."""
        RGB = namedtuple("RGB", ["r", "g"])
        # Point and RGB are different namedtuples with same field count
        # _treedefs_compatible checks `a.typ is not b.typ` → mismatch
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: x + y, Point(1, 2), RGB(10, 20))

    def test_different_arities_namedtuple_mismatch(self):
        """namedtuples with different field counts should mismatch."""
        RGB3 = namedtuple("RGB3", ["r", "g", "b"])
        # Point has 2 fields, RGB3 has 3 → length mismatch
        with pytest.raises(ValueError, match="structure mismatch"):
            _pure_tree_map(lambda x, y: x + y, Point(1, 2), RGB3(10, 20, 30))

    def test_map_over_dict_with_empty_list_value(self):
        tree = {"a": [], "b": [1, 2]}
        result = _pure_tree_map(lambda x: x * 2, tree)
        assert result == {"a": [], "b": [2, 4]}

    def test_map_preserves_empty_containers(self):
        for tree in [[], (), {}, None]:
            result = _pure_tree_map(lambda x: x, tree)
            assert result == tree

    def test_map_preserves_namedtuple_type(self):
        """namedtuple type must be preserved (not demoted to plain tuple)."""
        result = _pure_tree_map(lambda x: x * 2, Point(3, 4))
        assert isinstance(result, Point)
        assert result == Point(6, 8)

    def test_map_with_dict_same_keys_different_insertion_order(self):
        t1 = {"b": 2, "a": 1}
        t2 = {"a": 10, "b": 20}
        result = _pure_tree_map(lambda x, y: x + y, t1, t2)
        assert result == {"b": 22, "a": 11}

    def test_map_propagates_leaf_exception(self):
        """Exception raised by f should propagate with context."""

        def boom(x):
            raise RuntimeError("leaf error")

        with pytest.raises(RuntimeError, match="leaf error"):
            _pure_tree_map(boom, {"a": 1})


class TestJAXComparisonTreeMap:
    """Directly compare tree_map behaviour with JAX.
    These tests are skipped if JAX is not available."""

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax")

    def test_tree_map_basic_matches_jax(self):
        tree = {"a": [1, 2], "b": 3}
        py_result = _pure_tree_map(lambda x: x * 2, tree)
        jax_result = tree_util.tree_map(lambda x: x * 2, tree)
        assert py_result == jax_result

    def test_tree_map_two_args_matches_jax(self):
        t1 = {"a": [1, 2], "b": [3]}
        t2 = {"a": [10, 20], "b": [30]}
        py_result = _pure_tree_map(lambda x, y: x + y, t1, t2)
        jax_result = tree_util.tree_map(lambda x, y: x + y, t1, t2)
        assert py_result == jax_result

    def test_tree_map_over_none_matches_jax(self):
        py_result = _pure_tree_map(lambda x: x + 1, None)
        jax_result = tree_util.tree_map(lambda x: x + 1, None)
        assert py_result == jax_result

    def test_tree_map_rejects_mismatch_like_jax(self):
        # dict vs scalar: different input from the pure test
        with pytest.raises(ValueError):
            tree_util.tree_map(lambda x, y: (x, y), {"x": [3, 4]}, "bad")
        with pytest.raises(ValueError):
            _pure_tree_map(lambda x, y: (x, y), {"x": [3, 4]}, "bad")

    def test_tree_map_rejects_none_vs_scalar_like_jax(self):
        with pytest.raises((ValueError, TypeError)):
            tree_util.tree_map(lambda x, y: (x, y), None, 99)
        with pytest.raises(ValueError):
            _pure_tree_map(lambda x, y: (x, y), None, 99)


class TestJAXComparisonEmptyStructures:
    """Compare empty structure handling with JAX."""

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax")

    def test_empty_list_matches_jax(self):
        jax_leaves, _ = tree_util.tree_flatten([])
        pure_leaves, _ = _pure_tree_flatten([])
        assert jax_leaves == pure_leaves == []

    def test_empty_dict_matches_jax(self):
        jax_leaves, _ = tree_util.tree_flatten({})
        pure_leaves, _ = _pure_tree_flatten({})
        assert jax_leaves == pure_leaves == []

    def test_roundtrip_empty_list(self):
        jax_leaves, jax_td = tree_util.tree_flatten([])
        pure_leaves, pure_td = _pure_tree_flatten([])
        assert tree_util.tree_unflatten(jax_td, jax_leaves) == _pure_tree_unflatten(
            pure_td, pure_leaves
        )


# ---------------------------------------------------------------------------
# JAX comparison tests (requires jax)
# ---------------------------------------------------------------------------


class TestJAXComparison:
    """Directly compare behaviour with JAX ``tree_util``.
    These tests are skipped if JAX is not available."""

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax")

    def test_flatten_dict_key_order_matches_jax(self):
        tree = {"b": 2, "a": 1}
        jax_leaves, _ = tree_util.tree_flatten(tree)
        pure_leaves, _ = _pure_tree_flatten(tree)
        assert jax_leaves == pure_leaves

    def test_flatten_none_matches_jax(self):
        jax_leaves, _ = tree_util.tree_flatten(None)
        pure_leaves, _ = _pure_tree_flatten(None)
        assert jax_leaves == pure_leaves == []

    def test_unflatten_extra_leaves_matches_jax(self):
        _, jax_td = tree_util.tree_flatten({"a": 1})
        _, pure_td = _pure_tree_flatten({"a": 1})

        with pytest.raises(ValueError):
            tree_util.tree_unflatten(jax_td, [10, 11])
        with pytest.raises(ValueError):
            _pure_tree_unflatten(pure_td, [10, 11])

    def test_unflatten_too_few_leaves_matches_jax(self):
        _, jax_td = tree_util.tree_flatten({"a": 1, "b": 2})
        _, pure_td = _pure_tree_flatten({"a": 1, "b": 2})

        with pytest.raises(ValueError):
            tree_util.tree_unflatten(jax_td, [10])
        with pytest.raises(ValueError):
            _pure_tree_unflatten(pure_td, [10])

    def test_roundtrip_nested_matches_jax(self):
        tree = {"a": [1, (2, 3)], "b": {"c": 4}}
        jax_leaves, jax_td = tree_util.tree_flatten(tree)
        pure_leaves, pure_td = _pure_tree_flatten(tree)
        assert jax_leaves == pure_leaves
        assert tree_util.tree_unflatten(jax_td, jax_leaves) == _pure_tree_unflatten(
            pure_td, pure_leaves
        )

    def test_flatten_nested_with_none_matches_jax(self):
        tree = {"a": None, "b": [None, 1], "c": 3}
        jax_leaves, _ = tree_util.tree_flatten(tree)
        pure_leaves, _ = _pure_tree_flatten(tree)
        assert jax_leaves == pure_leaves
