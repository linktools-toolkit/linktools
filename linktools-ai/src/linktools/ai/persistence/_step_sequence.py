#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native snapshot roots use the archive's immutable sequence contract."""

from collections.abc import Mapping, Sequence

from ._archive import ImmutableSequenceNode


def build_step_sequence(contents: Sequence[str]) -> tuple[str | None, tuple[tuple[str, ImmutableSequenceNode], ...]]:
    """Build a canonical root without loading or extending a mutable predecessor."""
    nodes: dict[str, ImmutableSequenceNode] = {}
    root = None
    for content_id in contents:
        first = None if root is None else nodes[root]
        second = None if first is None or first.next_forest is None else nodes[first.next_forest]
        node = ImmutableSequenceNode.extend(content_id, first, second)
        root = node.identity
        nodes[root] = node
    return root, tuple(nodes.items())


def step_sequence_contents(
    root: str | None, count: int, nodes: Mapping[str, ImmutableSequenceNode],
) -> tuple[str, ...]:
    """Validate the complete bounded forest and preserve occurrence ordering."""
    if type(count) is not int or count < 0 or (root is None) != (count == 0):
        raise ValueError("snapshot sequence root is invalid")
    values: list[str] = []
    observed: set[str] = set()
    first = root
    for size in ImmutableSequenceNode.forest_weights(count):
        if first is None or first not in nodes:
            raise ValueError("snapshot sequence is incomplete")
        head = nodes[first]
        pending = [(first, size)]
        while pending:
            identity, expected = pending.pop()
            node = nodes.get(identity)
            if node is None or node.identity != identity or node.tree_count != expected:
                raise ValueError("snapshot sequence is missing or corrupt")
            observed.add(identity)
            values.append(node.content_id)
            if expected > 1:
                if node.left is None or node.right is None:
                    raise ValueError("snapshot sequence is missing or corrupt")
                child_size = (expected - 1) // 2
                pending.extend(((node.right, child_size), (node.left, child_size)))
        first = head.next_forest
    if first is not None or len(values) != count or observed != set(nodes):
        raise ValueError("snapshot sequence is incomplete")
    return tuple(reversed(values))


__all__ = ["build_step_sequence", "step_sequence_contents"]
