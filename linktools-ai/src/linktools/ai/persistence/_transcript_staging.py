#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local transcript visibility before checkpoint materialization."""

import hashlib
from collections.abc import Iterable, Iterator, Sequence
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
        self._digest_state = None
        self._encoded_size = 2
        self._prefix_digest = None

    def __copy__(self) -> "MessageSnapshot":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "MessageSnapshot":
        return self

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

    def shared_prefix_length(self, other: "MessageSnapshot") -> int:
        """Count the common immutable prefix without inspecting message bodies."""
        left, right = list(reversed(self._roots)), list(reversed(other._roots))
        count = 0
        while left and right:
            first, second = left[-1], right[-1]
            if first is second:
                count += first.count
                left.pop()
                right.pop()
            elif first.count >= second.count and first.payload is None:
                left.pop()
                left.extend((first.right, first.left))
            elif second.payload is None:
                right.pop()
                right.extend((second.right, second.left))
            else:
                break
        return count

    def encoded_payloads(self) -> Iterator[bytes]:
        """Yield the already frozen canonical singleton arrays."""
        return self._payloads()

    def encoded_digest(self) -> tuple[str, int]:
        """Identify the canonical message array without rebuilding SDK values."""
        if self._digest_state is None:
            digest = hashlib.sha256(b"[")
            size = 2
            for index, payload in enumerate(self._payloads()):
                if index:
                    digest.update(b",")
                    size += 1
                fragment = memoryview(payload)[1:-1]
                digest.update(fragment)
                size += len(fragment)
            self._digest_state, self._encoded_size = digest, size
        digest = self._digest_state.copy()
        digest.update(b"]")
        return digest.hexdigest(), self._encoded_size

    def __iter__(self) -> Iterator[ModelMessage]:
        for payload in self._payloads():
            yield decode_model_messages(payload)[0]

    def __getitem__(self, index: int | slice) -> "ModelMessage | MessageSnapshot":
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return MessageSnapshot(tuple(self._leaf(index) for index in range(start, stop, step)))
            lo, hi = start, max(start, stop)
            if lo == 0 and hi == len(self):
                return self
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
            result = MessageSnapshot(tuple(selected))
            if lo == 0 and self._prefix_digest is not None and hi == self._prefix_digest[0]:
                _, result._digest_state, result._encoded_size = self._prefix_digest
            return result
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

    def append(self, prefix: MessageSnapshot, delta: Sequence[ModelMessage]) -> MessageSnapshot:
        """Append an explicit delta to owned immutable history, freezing only the delta.

        Mutable full-history inputs must use snapshot instead; a length or object
        identity cannot establish that their historical contents stayed unchanged.
        """
        if not isinstance(prefix, MessageSnapshot):
            raise TypeError("message append requires an immutable prefix")
        return self.concatenate(prefix, delta)

    def concatenate(self, *groups: Sequence[ModelMessage]) -> MessageSnapshot:
        """Join frozen sequences without decoding or copying their message bodies."""
        frozen = tuple(self.snapshot(group) for group in groups if len(group))
        if not frozen:
            return MessageSnapshot(())
        if len(frozen) == 1:
            return frozen[0]
        first, *tail = frozen
        roots = list(first._roots)
        for group in tail:
            pending = list(reversed(group._roots))
            while pending:
                node = pending.pop()
                if roots and roots[-1].count < node.count:
                    pending.extend((node.right, node.left))
                    continue
                while roots and roots[-1].count == node.count:
                    left = roots.pop()
                    key = (left, node)
                    branch = self._branches.get(key)
                    if branch is None:
                        branch = _MessageTree(left.count + node.count, left=left, right=node)
                        self._branches[key] = branch
                    node = branch
                roots.append(node)
        first.encoded_digest()
        result = MessageSnapshot(tuple(roots))
        digest = first._digest_state.copy()
        size = first._encoded_size
        for group in tail:
            for payload in group.encoded_payloads():
                digest.update(b",")
                fragment = memoryview(payload)[1:-1]
                digest.update(fragment)
                size += 1 + len(fragment)
        result._digest_state, result._encoded_size = digest, size
        # This proof is carried only by immutable bytes, never SDK object identity.
        # Keep one predecessor state, not a chain retaining all prior snapshots.
        result._prefix_digest = (len(first), first._digest_state, first._encoded_size)
        return result

    def snapshot(self, messages: Sequence[ModelMessage]) -> MessageSnapshot:
        if isinstance(messages, MessageSnapshot):
            return messages
        return self._snapshot_payloads(encode_model_messages((message,)) for message in messages)

    def snapshot_payloads(self, payloads: Sequence[bytes]) -> MessageSnapshot:
        """Validate frozen singleton messages without replacing omitted wire fields."""
        for payload in payloads:
            if type(payload) is not bytes or len(decode_model_messages(payload)) != 1:
                raise ValueError("frozen message payload must contain exactly one message")
        return self._snapshot_payloads(payloads)

    def _snapshot_payloads(self, payloads: Iterable[bytes]) -> MessageSnapshot:
        roots: list[_MessageTree] = []
        digest = hashlib.sha256(b"[")
        size, count = 2, 0
        prefix = None
        for payload in payloads:
            prefix = (count, digest.copy(), size)
            if count:
                digest.update(b",")
                size += 1
            fragment = memoryview(payload)[1:-1]
            digest.update(fragment)
            size += len(fragment)
            count += 1
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
        snapshot = MessageSnapshot(tuple(roots))
        snapshot._digest_state, snapshot._encoded_size = digest, size
        snapshot._prefix_digest = prefix
        return snapshot
