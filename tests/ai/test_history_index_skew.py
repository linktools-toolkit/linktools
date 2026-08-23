"""Skew-binary session prefix index conformance."""

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import ConversationHistoryIndexNodeRecord
from linktools.ai.runtime.state._history_index import (
    build_fork_index_node_from_roots,
    resolve_history_range_lazy,
)


def _build_chain(
    appends: list[int],
) -> tuple[
    dict[str, ConversationHistoryIndexNodeRecord],
    tuple[str, ...],
    str,
]:
    """Build a fork chain where each branch owns a local message suffix."""
    nodes: dict[str, ConversationHistoryIndexNodeRecord] = {}
    roots: list[str] = []
    for index in range(1, len(appends)):
        node = build_fork_index_node_from_roots(
            [nodes[root] for root in roots],
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
    return nodes, tuple(roots), f"h{len(appends) - 1}"


def _flat(appends: list[int]) -> list[str]:
    values: list[str] = []
    for index, count in enumerate(appends):
        values.extend([f"h{index}"] * count)
    return values


async def _resolve(
    nodes: dict[str, ConversationHistoryIndexNodeRecord],
    roots: tuple[str, ...],
    *,
    head: str,
    local_message_count: int,
    inherited_message_count: int,
    start: int,
    end: int,
) -> list[str]:
    async def load_node(node_id: str) -> ConversationHistoryIndexNodeRecord:
        return nodes[node_id]

    resolved = await resolve_history_range_lazy(
        [nodes[root] for root in roots],
        load_node,
        owner_history_id=head,
        local_message_count=local_message_count,
        inherited_message_count=inherited_message_count,
        range_start=start,
        range_end=end,
    )
    values: list[str] = []
    for item in resolved:
        values.extend(
            [item.segment.owner_history_id] * (item.local_end - item.local_start)
        )
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "appends",
    [
        [3, 1, 4, 1, 5, 9, 2, 6],
        [0, 2, 0, 3, 5, 0, 1, 1, 0, 7, 0, 2],
        [4, 4, 4, 4, 4, 4, 4, 4, 4],
        [7, 2],
        [0, 0, 0, 5],
        [5, 0, 0],
        [index % 5 for index in range(30)],
        [1] * 20,
    ],
)
async def test_resolver_matches_logical_order(appends: list[int]) -> None:
    nodes, roots, head = _build_chain(appends)
    total = sum(appends)
    flat = _flat(appends)
    local_now = appends[-1]
    inherited = total - local_now
    for start in range(0, total, 3):
        end = min(start + 7, total)
        resolved = await _resolve(
            nodes,
            roots,
            head=head,
            local_message_count=local_now,
            inherited_message_count=inherited,
            start=start,
            end=end,
        )
        assert resolved == flat[start:end]


@pytest.mark.asyncio
async def test_zero_message_fork_of_empty_forest_yields_no_node() -> None:
    nodes, roots, _head = _build_chain([0, 0])
    assert nodes == {}
    assert roots == ()


@pytest.mark.asyncio
async def test_zero_message_fork_shares_existing_prefix_head() -> None:
    nodes, roots, _head = _build_chain([3, 0])
    shared = build_fork_index_node_from_roots(
        [nodes[root] for root in roots],
        source_history_id="h1",
        source_local_message_count=0,
        source_local_history_item_count=0,
    )
    assert shared == roots[0]


@pytest.mark.asyncio
async def test_empty_source_fork_then_nonempty_uses_shared_head() -> None:
    appends = [0, 0, 0, 5]
    nodes, roots, head = _build_chain(appends)
    resolved = await _resolve(
        nodes,
        roots,
        head=head,
        local_message_count=5,
        inherited_message_count=0,
        start=0,
        end=5,
    )
    assert resolved == ["h3"] * 5


@pytest.mark.asyncio
async def test_non_empty_fork_adds_at_most_one_node() -> None:
    nodes, _roots, _head = _build_chain([4, 4, 4, 4])
    assert len(nodes) == 3


@pytest.mark.asyncio
async def test_merge_takes_equal_weight_first_two_roots() -> None:
    nodes, roots, _head = _build_chain([4, 4, 4, 4, 4])
    assert len(roots) == 2
    merged = nodes[roots[1]]
    assert merged.right_tree_id is not None
    assert merged.segment.owner_history_id == "h2"
    assert (
        merged.tree_segment_count
        == nodes[merged.left_tree_id].tree_segment_count
        + nodes[merged.right_tree_id].tree_segment_count
        + 1
    )


@pytest.mark.asyncio
async def test_range_beyond_total_rejected() -> None:
    nodes, roots, head = _build_chain([3, 1, 4])
    with pytest.raises(AIError) as raised:
        await _resolve(
            nodes,
            roots,
            head=head,
            local_message_count=4,
            inherited_message_count=4,
            start=0,
            end=9,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_empty_range_returns_nothing() -> None:
    nodes, roots, head = _build_chain([3, 1, 4])
    resolved = await _resolve(
        nodes,
        roots,
        head=head,
        local_message_count=4,
        inherited_message_count=4,
        start=5,
        end=5,
    )
    assert resolved == []
