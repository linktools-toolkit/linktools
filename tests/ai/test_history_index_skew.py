#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skew-binary session prefix index conformance."""

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._history_index import (
    HistoryIndexSnapshot,
    build_fork_index_node,
    resolve_history_range,
)


def _build_chain(appends: "list[int]") -> tuple[HistoryIndexSnapshot, str]:
    """Fork chain h0..hN; branch i owns appends[i] local messages."""
    nodes: dict = {}
    roots: list[str] = []
    for index in range(1, len(appends)):
        node = build_fork_index_node(
            HistoryIndexSnapshot(nodes, tuple(roots)),
            source_history_id=f"h{index - 1}",
            source_local_message_count=appends[index - 1],
            source_local_history_item_count=appends[index - 1] * 2,
        )
        if node is None or isinstance(node, str):
            continue
        nodes[node.node_id] = node
        if node.right_tree_id is not None:
            roots = [node.node_id, *roots[2:]]
        else:
            roots = [node.node_id, *roots]
    return HistoryIndexSnapshot(nodes, tuple(roots)), f"h{len(appends) - 1}"


def _flat(appends: "list[int]") -> "list[str]":
    values: list[str] = []
    for index, count in enumerate(appends):
        values.extend([f"h{index}"] * count)
    return values


def _resolved_names(resolved) -> "list[str]":
    values: list[str] = []
    for item in resolved:
        values.extend(
            [item.segment.owner_history_id] * (item.local_end - item.local_start)
        )
    return values


@pytest.mark.parametrize(
    "appends",
    [
        [3, 1, 4, 1, 5, 9, 2, 6],
        [0, 2, 0, 3, 5, 0, 1, 1, 0, 7, 0, 2],
        [4, 4, 4, 4, 4, 4, 4, 4, 4],
        [7, 2],
        [0, 0, 0, 5],
        [5, 0, 0],
        [i % 5 for i in range(30)],
        [1] * 20,
    ],
)
def test_resolver_matches_logical_order(appends: "list[int]") -> None:
    snapshot, head = _build_chain(appends)
    total = sum(appends)
    flat = _flat(appends)
    local_now = appends[-1]
    inherited = total - local_now
    for start in range(0, total, 3):
        end = min(start + 7, total)
        resolved = resolve_history_range(
            snapshot,
            owner_history_id=head,
            local_message_count=local_now,
            inherited_message_count=inherited,
            range_start=start,
            range_end=end,
        )
        assert _resolved_names(resolved) == flat[start:end]


def test_zero_message_fork_of_empty_forest_yields_no_node() -> None:
    snapshot = HistoryIndexSnapshot({}, ())
    shared = build_fork_index_node(
        snapshot,
        source_history_id="h0",
        source_local_message_count=0,
        source_local_history_item_count=0,
    )
    assert shared is None


def test_zero_message_fork_shares_existing_prefix_head() -> None:
    node = build_fork_index_node(
        HistoryIndexSnapshot({}, ()),
        source_history_id="h0",
        source_local_message_count=3,
        source_local_history_item_count=6,
    )
    assert not isinstance(node, str) and node is not None
    snapshot = HistoryIndexSnapshot({node.node_id: node}, (node.node_id,))
    shared = build_fork_index_node(
        snapshot,
        source_history_id="h1",
        source_local_message_count=0,
        source_local_history_item_count=0,
    )
    assert shared == node.node_id


def test_empty_source_fork_then_nonempty_uses_shared_head() -> None:
    appends = [0, 0, 0, 5]
    snapshot, head = _build_chain(appends)
    # all forks shared the empty prefix; only h3 has content
    resolved = resolve_history_range(
        snapshot,
        owner_history_id=head,
        local_message_count=5,
        inherited_message_count=0,
        range_start=0,
        range_end=5,
    )
    assert _resolved_names(resolved) == ["h3"] * 5


def test_non_empty_fork_adds_at_most_one_node() -> None:
    appends = [4, 4, 4, 4]
    snapshot, _ = _build_chain(appends)
    # one node per non-empty-source fork, none for the root branch itself
    assert len(snapshot.nodes) == len(appends) - 1


def test_merge_takes_equal_weight_first_two_roots() -> None:
    snapshot, _ = _build_chain([4, 4, 4, 4, 4])
    roots = snapshot.root_ids
    assert len(roots) == 2
    merged = snapshot.nodes[roots[1]]
    assert merged.right_tree_id is not None
    assert merged.segment.owner_history_id == "h2"
    assert (
        merged.tree_segment_count
        == snapshot.nodes[merged.left_tree_id].tree_segment_count
        + snapshot.nodes[merged.right_tree_id].tree_segment_count
        + 1
    )


def test_range_beyond_total_rejected() -> None:
    snapshot, head = _build_chain([3, 1, 4])
    with pytest.raises(AIError) as raised:
        resolve_history_range(
            snapshot,
            owner_history_id=head,
            local_message_count=4,
            inherited_message_count=4,
            range_start=0,
            range_end=9,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_empty_range_returns_nothing() -> None:
    snapshot, head = _build_chain([3, 1, 4])
    resolved = resolve_history_range(
        snapshot,
        owner_history_id=head,
        local_message_count=4,
        inherited_message_count=4,
        range_start=5,
        range_end=5,
    )
    assert resolved == ()
