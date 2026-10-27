#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local transcript visibility before checkpoint materialization."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from weakref import WeakValueDictionary

from pydantic_ai.messages import ModelMessage

from ..model.messages import decode_model_messages, encode_model_messages


@dataclass(frozen=True, slots=True)
class StagedTranscript:
    messages: Sequence[ModelMessage]
    pending: ModelMessage | None = None
    pending_keys: tuple[str, ...] = ()


__all__ = ["StagedTranscript", "MessageArchive", "MessageSnapshot"]


@dataclass(frozen=True, eq=False)
class _MessageTree:
    count: int
    payload: bytes | None = None
    left: "_MessageTree | None" = None
    right: "_MessageTree | None" = None


class MessageSnapshot(Sequence[ModelMessage]):
    """Immutable encoded messages; each access returns independent SDK values."""

    def __init__(self, roots: tuple[_MessageTree, ...]) -> None:
        self._roots = roots
        self._count = sum(root.count for root in roots)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MessageSnapshot):
            return (self._roots == other._roots or len(self) == len(other)
                    and all(left == right for left, right in zip(self._payloads(), other._payloads())))
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(left == right for left, right in zip(self, other))
        return NotImplemented

    def __len__(self) -> int:
        return self._count

    def _payloads(self) -> Iterator[bytes]:
        stack = list(reversed(self._roots))
        while stack:
            node = stack.pop()
            if node.payload is not None:
                yield node.payload
            else:
                stack.append(node.right)
                stack.append(node.left)

    def __iter__(self) -> Iterator[ModelMessage]:
        for payload in self._payloads():
            yield decode_model_messages(payload)[0]

    def __getitem__(self, index: int | slice) -> "ModelMessage | MessageSnapshot":
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return MessageSnapshot(tuple(self._leaf(index) for index in range(start, stop, step)))
            lo, hi = start, max(start, stop)
            selected: list[_MessageTree] = []
            offset = 0
            for root in self._roots:
                stack = [(root, offset)]
                while stack:
                    node, first = stack.pop()
                    if first >= hi or first + node.count <= lo:
                        continue
                    if lo <= first and first + node.count <= hi:
                        selected.append(node)
                    else:
                        stack.append((node.right, first + node.left.count))
                        stack.append((node.left, first))
                offset += root.count
            return MessageSnapshot(tuple(selected))
        return decode_model_messages(self._leaf(index).payload)[0]

    def _leaf(self, index: int) -> _MessageTree:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        offset = index
        for root in self._roots:
            if offset < root.count:
                node = root
                while node.payload is None:
                    if offset < node.left.count:
                        node = node.left
                    else:
                        offset -= node.left.count
                        node = node.right
                return node
            offset -= root.count
        raise IndexError(index)


class MessageArchive:
    """Process-local immutable content and shared balanced sequence nodes."""

    def __init__(self) -> None:
        self._leaves: WeakValueDictionary[bytes, _MessageTree] = WeakValueDictionary()
        self._branches: WeakValueDictionary[tuple[_MessageTree, _MessageTree], _MessageTree] = WeakValueDictionary()

    def snapshot(self, messages: Sequence[ModelMessage]) -> MessageSnapshot:
        if isinstance(messages, MessageSnapshot):
            return messages
        roots: list[_MessageTree] = []
        for message in messages:
            payload = encode_model_messages((message,))
            node = self._leaves.get(payload)
            if node is None:
                node = _MessageTree(1, payload)
                self._leaves[payload] = node
            while roots and roots[-1].count == node.count:
                left = roots.pop()
                key = (left, node)
                branch = self._branches.get(key)
                if branch is None:
                    branch = _MessageTree(left.count + node.count, left=left, right=node)
                    self._branches[key] = branch
                node = branch
            roots.append(node)
        return MessageSnapshot(tuple(roots))
