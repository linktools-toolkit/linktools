#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable references to canonical native message occurrence ranges."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core import JsonValue
from ..steps import validate_native_store_key, encode_native_store_key, decode_native_store_key


_MAX_COUNT = (1 << 63) - 1


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("native message digest is invalid")
    try:
        if bytes.fromhex(value).hex() != value:
            raise ValueError("native message digest is invalid")
    except ValueError as error:
        raise ValueError("native message digest is invalid") from error
    return value


def _count(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError("native message count is invalid")
    return value


@dataclass(frozen=True, slots=True)
class NativeMessageSlice:
    """A range of an immutable sequence, never a prunable snapshot row."""

    first: str
    count: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _digest(self.first)
        if not 0 <= _count(self.start) < _count(self.end) <= _count(self.count):
            raise ValueError("native message range is invalid")


def _validate_slices(slices: tuple[NativeMessageSlice, ...]) -> None:
    if type(slices) is not tuple or any(not isinstance(item, NativeMessageSlice) for item in slices):
        raise ValueError("native message slices must be immutable")
    _count(sum(item.end - item.start for item in slices))


def _slice_ranges(slices: tuple[NativeMessageSlice, ...], start: int, end: int) -> tuple[NativeMessageSlice, ...]:
    if not 0 <= _count(start) <= _count(end) <= sum(item.end - item.start for item in slices):
        raise ValueError("native message range is invalid")
    selected = []
    offset = 0
    for item in slices:
        size = item.end - item.start
        lo, hi = max(0, start - offset), min(size, end - offset)
        if lo < hi:
            selected.append(NativeMessageSlice(item.first, item.count, item.start + lo, item.start + hi))
        offset += size
        if offset >= end:
            break
    return tuple(selected)


def _encode_slices(slices: tuple[NativeMessageSlice, ...]) -> list[JsonValue]:
    return [{"first": item.first, "count": item.count, "start": item.start, "end": item.end} for item in slices]


def _decode_slices(value: object) -> tuple[NativeMessageSlice, ...]:
    if not isinstance(value, list):
        raise ValueError("native message range wire is invalid")
    slices = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"first", "count", "start", "end"}:
            raise ValueError("native message range wire is invalid")
        slices.append(NativeMessageSlice(item["first"], item["count"], item["start"], item["end"]))
    return tuple(slices)


@dataclass(frozen=True, slots=True)
class NativeMessageSequence:
    """A native owner's internal sequence slot, scoped by its containing row.

    This value is not a portable reference or proof that content exists. Only a
    native owner's trusted slot supplies its SQL context and store partition.
    Cross-owner records must retain NativeMessageRoot with the actual group ID.
    """

    slices: tuple[NativeMessageSlice, ...] = ()

    def __post_init__(self) -> None:
        _validate_slices(self.slices)

    @property
    def count(self) -> int:
        return sum(item.end - item.start for item in self.slices)

    def slice(self, start: int, end: int) -> "NativeMessageSequence":
        return NativeMessageSequence(_slice_ranges(self.slices, start, end))

    def to_json(self) -> dict[str, JsonValue]:
        """Encode only inside the native owner's declared sequence slot."""
        return {"type": "native_message_sequence", "slices": _encode_slices(self.slices)}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeMessageSequence":
        if (not isinstance(value, Mapping) or set(value) != {"type", "slices"}
                or value["type"] != "native_message_sequence"):
            raise ValueError("native message sequence wire is invalid")
        return cls(_decode_slices(value["slices"]))


@dataclass(frozen=True, slots=True)
class NativeMessageRoot:
    group_id: str
    store_key: bytes = field(repr=False)
    slices: tuple[NativeMessageSlice, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("native message group is required")
        validate_native_store_key(self.store_key)
        _validate_slices(self.slices)

    @property
    def count(self) -> int:
        return sum(item.end - item.start for item in self.slices)

    @property
    def sequence(self) -> NativeMessageSequence:
        """Expose the same immutable ranges without changing their encoding."""
        return NativeMessageSequence(self.slices)

    def to_json(self) -> dict[str, JsonValue]:
        """Encode a root only in the owning record's declared reference slot."""
        return {
            "type": "native_message_root", "group_id": self.group_id,
            "store_key": encode_native_store_key(self.store_key), "slices": _encode_slices(self.slices),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeMessageRoot":
        """Decode a trusted root slot, never discover roots in business JSON."""
        if (not isinstance(value, Mapping)
                or set(value) != {"type", "group_id", "store_key", "slices"}
                or value["type"] != "native_message_root"):
            raise ValueError("native message root wire is invalid")
        return cls(value["group_id"], decode_native_store_key(value["store_key"]), _decode_slices(value["slices"]))

    def slice(self, start: int, end: int) -> "NativeMessageRoot":
        """Retain descriptors in O(r); repeated snapshots still cost O(s*r)."""
        return NativeMessageRoot(self.group_id, self.store_key, _slice_ranges(self.slices, start, end))


@dataclass(frozen=True, slots=True)
class NativeTextReference:
    """One immutable response text projected into a canonical JSON result."""
    root: NativeMessageRoot
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native text reference requires one message")
        _digest(self.digest)
        _count(self.size)

    def to_json(self) -> dict[str, JsonValue]:
        return {"$native_text": self.root.to_json(), "digest": self.digest, "size": self.size}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeTextReference":
        if not isinstance(value, Mapping) or set(value) != {"$native_text", "digest", "size"}:
            raise ValueError("native text reference wire is invalid")
        return cls(NativeMessageRoot.from_json(value["$native_text"]), value["digest"], value["size"])


@dataclass(frozen=True, slots=True)
class NativeToolResultReference:
    """Project one published return into an exact original operation receipt."""

    root: NativeMessageRoot
    part_index: int
    tool_name: str
    tool_call_id: str
    template: str
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native tool result requires one message")
        _count(self.part_index)
        _digest(self.digest)
        _count(self.size)
        if any(not isinstance(value, str) or not value for value in
               (self.tool_name, self.tool_call_id, self.template)):
            raise ValueError("native tool result identity is invalid")

    def to_json(self) -> dict[str, JsonValue]:
        return {"$native_tool_result": self.root.to_json(), "part_index": self.part_index,
                "tool_name": self.tool_name, "tool_call_id": self.tool_call_id,
                "template": self.template, "digest": self.digest, "size": self.size}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeToolResultReference":
        if not isinstance(value, Mapping) or set(value) != {
                "$native_tool_result", "part_index", "tool_name", "tool_call_id", "template", "digest", "size"}:
            raise ValueError("native tool result wire is invalid")
        return cls(NativeMessageRoot.from_json(value["$native_tool_result"]), value["part_index"],
                   value["tool_name"], value["tool_call_id"], value["template"], value["digest"], value["size"])


__all__ = ["NativeToolResultReference", "NativeTextReference", "NativeMessageRoot", "NativeMessageSequence", "NativeMessageSlice"]
