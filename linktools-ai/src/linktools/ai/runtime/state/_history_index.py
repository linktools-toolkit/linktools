#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skew-binary session prefix index construction and range resolution."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import uuid4

from ...errors import AIError, ErrorCode
from ._contracts import (
    ConversationHistoryIndexNodeRecord,
    ConversationHistorySegmentRef,
)

__all__ = [
    "HistoryIndexSnapshot",
    "build_fork_index_node",
    "resolve_history_range",
]


@dataclass(frozen=True, slots=True)
class ResolvedSegment:
    """One contributing segment with its resolved local window."""

    segment: ConversationHistorySegmentRef
    local_start: int
    local_end: int


@dataclass(frozen=True, slots=True)
class HistoryIndexSnapshot:
    """Read-only view of the forest roots addressed by one prefix head."""

    nodes: Mapping[str, ConversationHistoryIndexNodeRecord]
    root_ids: tuple[str, ...]


def _segment(
    owner_history_id: str,
    through_messages: int,
    through_items: int,
) -> ConversationHistorySegmentRef:
    return ConversationHistorySegmentRef(
        owner_history_id,
        through_messages,
        through_items,
    )


def build_fork_index_node(
    source: HistoryIndexSnapshot,
    *,
    source_history_id: str,
    source_local_message_count: int,
    source_local_history_item_count: int,
) -> "ConversationHistoryIndexNodeRecord | str | None":
    """O(1) skew cons over the source forest.

    Returns a new merged/leaf node when the source local prefix is non-empty,
    the shared prefix head id when it is empty but the source already has an
    inherited forest (zero new nodes), or ``None`` when the source has no
    local content and no forest at all (nothing to inherit).
    """
    if source_local_message_count < 0 or source_local_history_item_count < 0:
        raise ValueError("source local counts cannot be negative")
    if source_local_message_count == 0 and source_local_history_item_count == 0:
        if source.root_ids:
            return source.root_ids[0]
        return None
    frozen = _segment(
        source_history_id,
        source_local_message_count,
        source_local_history_item_count,
    )
    if len(source.root_ids) >= 2:
        first = source.nodes[source.root_ids[0]]
        second = source.nodes[source.root_ids[1]]
        if first.tree_segment_count == second.tree_segment_count:
            return ConversationHistoryIndexNodeRecord(
                uuid4().hex,
                frozen,
                first.tree_segment_count + second.tree_segment_count + 1,
                first.tree_message_count
                + second.tree_message_count
                + source_local_message_count,
                first.tree_history_item_count
                + second.tree_history_item_count
                + source_local_history_item_count,
                first.node_id,
                second.node_id,
                second.next_forest_id,
            )
    node = ConversationHistoryIndexNodeRecord(
        uuid4().hex,
        frozen,
        1,
        source_local_message_count,
        source_local_history_item_count,
        None,
        None,
        source.root_ids[0] if source.root_ids else None,
    )
    return node


def resolve_history_range(
    index: HistoryIndexSnapshot,
    *,
    owner_history_id: str,
    local_message_count: int,
    inherited_message_count: int,
    range_start: int,
    range_end: int,
) -> tuple[ResolvedSegment, ...]:
    """Resolve a logical forward range into contributing segment windows.

    The forest is ordered newest-inherited-segment-first; the resolver walks
    it with weighted interval pruning so only intersecting trees are visited.
    """
    if local_message_count < 0 or inherited_message_count < 0:
        raise ValueError("history counts cannot be negative")
    if range_start < 0 or range_end < range_start:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    total = inherited_message_count + local_message_count
    if range_end > total:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if range_end == range_start:
        return ()
    resolved: list[ResolvedSegment] = []
    inherited_start = range_start
    inherited_end = min(range_end, inherited_message_count)
    if inherited_end > inherited_start:
        forest_total = sum(
            index.nodes[root_id].tree_message_count for root_id in index.root_ids
        )
        if forest_total != inherited_message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from_newest_start = inherited_message_count - inherited_end
        from_newest_end = inherited_message_count - inherited_start
        remaining_start = from_newest_start
        remaining_end = from_newest_end
        for root_id in index.root_ids:
            root = index.nodes.get(root_id)
            if root is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            root_span = root.tree_message_count
            if remaining_start >= root_span:
                remaining_start -= root_span
                remaining_end -= root_span
                continue
            tree_output: list[tuple[ConversationHistorySegmentRef, int, int]] = []
            _collect_tree(
                index.nodes,
                root,
                remaining_start,
                min(remaining_end, root_span),
                tree_output,
            )
            resolved.extend(
                ResolvedSegment(segment, start, end)
                for segment, start, end in tree_output
            )
            if remaining_end <= root_span:
                break
            remaining_start = 0
            remaining_end -= root_span
        resolved.reverse()
    local_start = max(range_start - inherited_message_count, 0)
    local_end = min(range_end - inherited_message_count, local_message_count)
    if local_end > local_start:
        resolved.append(
            ResolvedSegment(
                _segment(owner_history_id, local_message_count, 0),
                local_start,
                local_end,
            )
        )
    return tuple(resolved)


def _collect_tree(
    nodes: Mapping[str, ConversationHistoryIndexNodeRecord],
    node: ConversationHistoryIndexNodeRecord,
    window_start: int,
    window_end: int,
    output: list[tuple[ConversationHistorySegmentRef, int, int]],
) -> None:
    """Newest-first recursion: own segment, then left (newer) subtree, then right.

    A merged node covers ``[own segment][first root][second root]`` because
    the first forest root holds the newer inherited segments.
    """
    if window_end <= window_start:
        return
    left = None if node.left_tree_id is None else nodes.get(node.left_tree_id)
    right = None if node.right_tree_id is None else nodes.get(node.right_tree_id)
    if node.left_tree_id is not None and left is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if node.right_tree_id is not None and right is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    left_span = 0 if left is None else left.tree_message_count
    right_span = 0 if right is None else right.tree_message_count
    own_count = node.segment.through_local_message_count
    left_start = own_count
    right_start = own_count + left_span
    if window_start >= own_count + left_span + right_span:
        return
    if window_start < own_count:
        own_end = min(window_end, own_count)
        output.append((node.segment, window_start, own_end))
    if left is not None and window_end > left_start and window_start < left_start + left_span:
        _collect_tree(
            nodes,
            left,
            max(window_start - left_start, 0),
            min(window_end - left_start, left_span),
            output,
        )
    if right is not None and window_end > right_start:
        _collect_tree(
            nodes,
            right,
            max(window_start - right_start, 0),
            min(window_end - right_start, right_span),
            output,
        )
