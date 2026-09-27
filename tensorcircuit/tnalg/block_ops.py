"""
Static joins and batched contractions of charge-labelled tensor blocks.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

import jax
import jax.numpy as jnp

Metadata = Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]
Blocks = Tuple[Any, ...]


@dataclass(frozen=True)
class BlockContraction:
    """
    One binary contraction with reusable block joins and BLAS shape groups.

    :ivar output: Output block keys and true output shapes.
    :ivar axes: Contracted axes for the left and right operands.
    :ivar permutation: Permutation into the requested output order.
    :ivar groups: Prepared shape-compatible contraction groups.
    """

    output: Metadata
    axes: Tuple[Tuple[int, ...], Tuple[int, ...]]
    permutation: Tuple[int, ...]
    groups: Tuple[Any, ...]

    def __call__(self, left: Blocks, right: Blocks) -> Blocks:
        if len(left) == len(right) == len(self.output[0]) == 1:
            return (
                jnp.transpose(
                    jnp.tensordot(left[0], right[0], self.axes), self.permutation
                ),
            )
        result: list[Any] = [None] * len(self.output[0])
        for shape, outputs, terms in self.groups:
            value = jnp.zeros(
                (len(outputs),) + shape, dtype=jnp.result_type(left[0], right[0])
            )
            for left_unique, right_unique, left_rows, right_rows, destinations in terms:
                a = jnp.stack(tuple(left[i] for i in left_unique))[
                    jnp.asarray(left_rows)
                ]
                b = jnp.stack(tuple(right[i] for i in right_unique))[
                    jnp.asarray(right_rows)
                ]
                values = jax.vmap(
                    lambda x, y: jnp.transpose(
                        jnp.tensordot(x, y, self.axes), self.permutation
                    )
                )(a, b)
                value = value.at[jnp.asarray(destinations)].add(values)
            for row, output in enumerate(outputs):
                result[output] = value[
                    (row,) + tuple(slice(0, d) for d in self.output[1][output])
                ]
        return tuple(result)


@lru_cache(maxsize=4096)
def prepare_contraction(
    equation: str, left: Metadata, right: Metadata
) -> BlockContraction:
    """
    Join charges and batch contractions with identical true input/output shapes.

    :param equation: Einstein contraction equation with one output arrow.
    :type equation: str
    :param left: Keys and shapes of the left operand blocks.
    :type left: Metadata
    :param right: Keys and shapes of the right operand blocks.
    :type right: Metadata
    :return: Cached structural contraction plan.
    :rtype: BlockContraction
    """
    inputs, output_labels = equation.split("->")
    left_labels, right_labels = inputs.split(",")
    common = tuple(label for label in left_labels if label in right_labels)
    if any(label in output_labels for label in common):
        raise ValueError("block contraction requires shared indices to be contracted")
    axes = (
        tuple(left_labels.index(label) for label in common),
        tuple(right_labels.index(label) for label in common),
    )
    left_free = tuple(i for i in range(len(left_labels)) if i not in axes[0])
    right_free = tuple(i for i in range(len(right_labels)) if i not in axes[1])
    free_labels = tuple(left_labels[i] for i in left_free) + tuple(
        right_labels[i] for i in right_free
    )
    permutation = tuple(free_labels.index(label) for label in output_labels)
    right_lookup: dict[Any, list[int]] = {}
    for position, key in enumerate(right[0]):
        right_lookup.setdefault(tuple(key[i] for i in axes[1]), []).append(position)
    output_shapes: dict[Any, Any] = {}
    terms = []
    for a, key in enumerate(left[0]):
        for b in right_lookup.get(tuple(key[i] for i in axes[0]), ()):
            free_key = tuple(key[i] for i in left_free) + tuple(
                right[0][b][i] for i in right_free
            )
            free_shape = tuple(left[1][a][i] for i in left_free) + tuple(
                right[1][b][i] for i in right_free
            )
            output_key = tuple(free_key[i] for i in permutation)
            shape = tuple(free_shape[i] for i in permutation)
            output_shapes[output_key] = shape
            terms.append((output_key, a, b))
    keys = tuple(sorted(output_shapes))
    positions = {key: i for i, key in enumerate(keys)}
    shape_groups: dict[Any, list[int]] = {}
    for key in keys:
        shape_groups.setdefault(output_shapes[key], []).append(positions[key])
    groups = []
    for shape, outputs in shape_groups.items():
        destinations = {output: row for row, output in enumerate(outputs)}
        inputs_by_shape: dict[Any, list[Any]] = {}
        for key, a, b in terms:
            output = positions[key]
            if output in destinations:
                inputs_by_shape.setdefault((left[1][a], right[1][b]), []).append(
                    (a, b, destinations[output])
                )
        prepared = []
        for rows in inputs_by_shape.values():
            left_unique = tuple(dict.fromkeys(row[0] for row in rows))
            right_unique = tuple(dict.fromkeys(row[1] for row in rows))
            prepared.append(
                (
                    left_unique,
                    right_unique,
                    tuple(left_unique.index(row[0]) for row in rows),
                    tuple(right_unique.index(row[1]) for row in rows),
                    tuple(row[2] for row in rows),
                )
            )
        groups.append((shape, tuple(outputs), tuple(prepared)))
    return BlockContraction(
        (keys, tuple(output_shapes[key] for key in keys)),
        axes,
        permutation,
        tuple(groups),
    )


def align_blocks(
    metadata: Metadata, blocks: Blocks, target: Metadata, dtype: Any
) -> Blocks:
    """
    Supply structural zeros for output blocks absent from a sparse operator.

    :param metadata: Keys and shapes represented by ``blocks``.
    :type metadata: Metadata
    :param blocks: Existing sparse output blocks.
    :type blocks: Blocks
    :param target: Required output keys and shapes.
    :type target: Metadata
    :param dtype: Dtype used for structural zero blocks.
    :type dtype: Any
    :return: Blocks aligned to the target metadata.
    :rtype: Blocks
    """
    positions = {key: i for i, key in enumerate(metadata[0])}
    return tuple(
        blocks[positions[key]] if key in positions else jnp.zeros(shape, dtype=dtype)
        for key, shape in zip(*target)
    )
