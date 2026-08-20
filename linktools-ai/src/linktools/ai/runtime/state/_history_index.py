#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skew-binary session prefix index construction and range resolution."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from uuid import uuid4

from ...errors import AIError, ErrorCode
from ._contracts import (
    ConversationHistoryIndexNodeRecord,
    ConversationHistorySegmentRef,
)

__all__ = [
    "build_fork_index_node_from_roots",
    "resolve_history_range_lazy",
    "resolve_history_item_range_lazy",
]


@dataclass(frozen=True, slots=True)
class ResolvedSegment:
    """One contributing segment with its resolved local window."""

    segment: ConversationHistorySegmentRef
    local_start: int
    local_end: int


async def resolve_history_range_lazy(
    roots: Sequence[ConversationHistoryIndexNodeRecord],
    load_node: Callable[[str], Awaitable[ConversationHistoryIndexNodeRecord]],
    *,
    owner_history_id: str,
    local_message_count: int,
    inherited_message_count: int,
    range_start: int,
    range_end: int,
) -> tuple[ResolvedSegment, ...]:
    """Resolve a message range while loading only intersecting tree nodes."""
    if local_message_count < 0 or inherited_message_count < 0:
        raise ValueError("history counts cannot be negative")
    if range_start < 0 or range_end < range_start:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    total = inherited_message_count + local_message_count
    if range_end > total:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if range_end == range_start:
        return ()
    root_values = tuple(roots)
    if sum(node.tree_message_count for node in root_values) != inherited_message_count:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    resolved: list[ResolvedSegment] = []
    inherited_start = range_start
    inherited_end = min(range_end, inherited_message_count)
    if inherited_end > inherited_start:
        remaining_start = inherited_message_count - inherited_end
        remaining_end = inherited_message_count - inherited_start
        for root in root_values:
            root_span = root.tree_message_count
            if remaining_start >= root_span:
                remaining_start -= root_span
                remaining_end -= root_span
                continue
            tree_output: list[tuple[ConversationHistorySegmentRef, int, int]] = []
            await _collect_tree_lazy(
                root,
                remaining_start,
                min(remaining_end, root_span),
                load_node,
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


async def resolve_history_item_range_lazy(
    roots: Sequence[ConversationHistoryIndexNodeRecord],
    load_node: Callable[[str], Awaitable[ConversationHistoryIndexNodeRecord]],
    *,
    owner_history_id: str,
    local_history_item_count: int,
    inherited_history_item_count: int,
    range_start: int,
    range_end: int,
) -> tuple[ResolvedSegment, ...]:
    """Resolve a Session-history item range using weighted index counters."""
    if local_history_item_count < 0 or inherited_history_item_count < 0:
        raise ValueError("history item counts cannot be negative")
    if range_start < 0 or range_end < range_start:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    total = inherited_history_item_count + local_history_item_count
    if range_end > total:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if range_end == range_start:
        return ()
    root_values = tuple(roots)
    if (
        sum(node.tree_history_item_count for node in root_values)
        != inherited_history_item_count
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    resolved: list[ResolvedSegment] = []
    inherited_start = range_start
    inherited_end = min(range_end, inherited_history_item_count)
    if inherited_end > inherited_start:
        remaining_start = inherited_history_item_count - inherited_end
        remaining_end = inherited_history_item_count - inherited_start
        for root in root_values:
            root_span = root.tree_history_item_count
            if remaining_start >= root_span:
                remaining_start -= root_span
                remaining_end -= root_span
                continue
            tree_output: list[tuple[ConversationHistorySegmentRef, int, int]] = []
            await _collect_tree_items_lazy(
                root,
                remaining_start,
                min(remaining_end, root_span),
                load_node,
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
    local_start = max(range_start - inherited_history_item_count, 0)
    local_end = min(
        range_end - inherited_history_item_count,
        local_history_item_count,
    )
    if local_end > local_start:
        resolved.append(
            ResolvedSegment(
                _segment(owner_history_id, 0, local_history_item_count),
                local_start,
                local_end,
            )
        )
    return tuple(resolved)


async def _collect_tree_lazy(
    node: ConversationHistoryIndexNodeRecord,
    window_start: int,
    window_end: int,
    load_node: Callable[[str], Awaitable[ConversationHistoryIndexNodeRecord]],
    output: list[tuple[ConversationHistorySegmentRef, int, int]],
) -> None:
    if window_end <= window_start:
        return
    own_count = node.segment.through_local_message_count
    if window_end <= own_count:
        if window_start < own_count:
            output.append((node.segment, window_start, min(window_end, own_count)))
        return
    left = None
    if node.left_tree_id is not None:
        left = await load_node(node.left_tree_id)
    left_span = 0 if left is None else left.tree_message_count
    left_start = own_count
    right_start = own_count + left_span
    right = None
    if node.right_tree_id is not None and window_end > right_start:
        right = await load_node(node.right_tree_id)
    right_span = 0 if right is None else right.tree_message_count
    if window_start >= own_count + left_span + right_span:
        return
    if window_start < own_count:
        output.append((node.segment, window_start, min(window_end, own_count)))
    if left is not None and window_end > left_start and window_start < left_start + left_span:
        await _collect_tree_lazy(
            left,
            max(window_start - left_start, 0),
            min(window_end - left_start, left_span),
            load_node,
            output,
        )
    if right is not None and window_end > right_start:
        await _collect_tree_lazy(
            right,
            max(window_start - right_start, 0),
            min(window_end - right_start, right_span),
            load_node,
            output,
        )


async def _collect_tree_items_lazy(
    node: ConversationHistoryIndexNodeRecord,
    window_start: int,
    window_end: int,
    load_node: Callable[[str], Awaitable[ConversationHistoryIndexNodeRecord]],
    output: list[tuple[ConversationHistorySegmentRef, int, int]],
) -> None:
    if window_end <= window_start:
        return
    own_count = node.segment.through_local_history_item_count
    if window_end <= own_count:
        if window_start < own_count:
            output.append((node.segment, window_start, min(window_end, own_count)))
        return
    left = None
    if node.left_tree_id is not None:
        left = await load_node(node.left_tree_id)
    left_span = 0 if left is None else left.tree_history_item_count
    left_start = own_count
    right_start = own_count + left_span
    right = None
    if node.right_tree_id is not None and window_end > right_start:
        right = await load_node(node.right_tree_id)
    right_span = 0 if right is None else right.tree_history_item_count
    if window_start >= own_count + left_span + right_span:
        return
    if window_start < own_count:
        output.append((node.segment, window_start, min(window_end, own_count)))
    if left is not None and window_end > left_start and window_start < left_start + left_span:
        await _collect_tree_items_lazy(
            left,
            max(window_start - left_start, 0),
            min(window_end - left_start, left_span),
            load_node,
            output,
        )
    if right is not None and window_end > right_start:
        await _collect_tree_items_lazy(
            right,
            max(window_start - right_start, 0),
            min(window_end - right_start, right_span),
            load_node,
            output,
        )


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


def build_fork_index_node_from_roots(
    roots: Sequence[ConversationHistoryIndexNodeRecord],
    *,
    source_history_id: str,
    source_local_message_count: int,
    source_local_history_item_count: int,
) -> "ConversationHistoryIndexNodeRecord | str | None":
    """Build one skew node from at most the first two forest roots."""
    if source_local_message_count < 0 or source_local_history_item_count < 0:
        raise ValueError("source local counts cannot be negative")
    if source_local_message_count == 0 and source_local_history_item_count == 0:
        return roots[0].node_id if roots else None
    frozen = _segment(
        source_history_id,
        source_local_message_count,
        source_local_history_item_count,
    )
    if len(roots) >= 2 and roots[0].tree_segment_count == roots[1].tree_segment_count:
        first, second = roots[:2]
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
    return ConversationHistoryIndexNodeRecord(
        uuid4().hex,
        frozen,
        1,
        source_local_message_count,
        source_local_history_item_count,
        None,
        None,
        roots[0].node_id if roots else None,
    )
