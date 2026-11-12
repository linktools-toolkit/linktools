#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable references to canonical native message occurrence ranges."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from ..core import JsonValue
from ..storage import ObjectRef
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
    """A fixed response part projected into the exact accepted JSON output."""
    root: NativeMessageRoot
    projection: str = "text"
    part_index: int = 0
    tool_name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native output reference requires one message")
        _count(self.part_index)
        if self.projection not in {"text", "assistant-text-output", "tool-call-output"}:
            raise ValueError("unknown native output projection")
        if self.projection == "tool-call-output":
            if any(not isinstance(value, str) or not value for value in (self.tool_name, self.tool_call_id)):
                raise ValueError("native output tool identity is required")
        elif self.tool_name is not None or self.tool_call_id is not None:
            raise ValueError("text projection cannot carry tool identity")

    def to_json(self) -> dict[str, JsonValue]:
        result = {"$native_text": self.root.to_json()}
        if self.projection != "text":
            result["projection"] = self.projection
        if self.part_index:
            result["part_index"] = self.part_index
        if self.tool_name is not None:
            result.update(tool_name=self.tool_name, tool_call_id=self.tool_call_id)
        return result

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeTextReference":
        if (not isinstance(value, Mapping) or "$native_text" not in value
                or set(value) - {"$native_text", "projection", "part_index", "tool_name", "tool_call_id"}):
            raise ValueError("native output reference wire is invalid")
        return cls(NativeMessageRoot.from_json(value["$native_text"]), value.get("projection", "text"),
                   value.get("part_index", 0), value.get("tool_name"), value.get("tool_call_id"))


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
    attachment_format: str | None = None
    attachments: tuple[tuple[tuple[str | int, ...], ObjectRef], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native tool result requires one message")
        _count(self.part_index)
        _digest(self.digest)
        _count(self.size)
        if any(not isinstance(value, str) or not value for value in
               (self.tool_name, self.tool_call_id, self.template)):
            raise ValueError("native tool result identity is invalid")

        if self.attachment_format is None:
            if self.attachments:
                raise ValueError("attachment format is required")
        elif self.attachment_format != "model-messages" or not self.attachments:
            raise ValueError("invalid tool receipt attachment format")
        if not isinstance(self.attachments, tuple):
            raise ValueError("attachment entries must be immutable")
        paths = set()
        for path, reference in self.attachments:
            if (not isinstance(path, tuple) or not path or path in paths
                    or any(not isinstance(item, str) and (type(item) is not int or item < 0) for item in path)
                    or not isinstance(reference, ObjectRef)):
                raise ValueError("invalid tool receipt attachment entry")
            paths.add(path)

    def object_refs(self) -> tuple[ObjectRef, ...]:
        return tuple(reference for _path, reference in self.attachments)

    def attachment_entries(self) -> list[dict[str, JsonValue]]:
        return [{"path": list(path), "ref": {"store_id": ref.store_id, "key": ref.key,
                "digest": ref.digest, "size": ref.size}} for path, ref in self.attachments]

    def map_object_refs(self, mapper: Callable[[ObjectRef], ObjectRef]) -> "NativeToolResultReference":
        entries = []
        for path, ref in self.attachments:
            target = mapper(ref)
            if (target.digest, target.size) != (ref.digest, ref.size):
                raise ValueError("relocated receipt attachments must keep their bytes")
            entries.append((path, target))
        return replace(self, attachments=tuple(entries))

    def to_json(self) -> dict[str, JsonValue]:
        result = {"$native_tool_result": self.root.to_json(), "part_index": self.part_index,
                  "tool_name": self.tool_name, "tool_call_id": self.tool_call_id,
                  "template": self.template, "digest": self.digest, "size": self.size}
        if self.attachment_format is not None:
            result.update(attachment_format=self.attachment_format, attachments=self.attachment_entries())
        return result

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeToolResultReference":
        required = {"$native_tool_result", "part_index", "tool_name", "tool_call_id", "template", "digest", "size"}
        if (not isinstance(value, Mapping) or set(value) not in (required, required | {"attachment_format", "attachments"})):
            raise ValueError("native tool result wire is invalid")
        entries = value.get("attachments", [])
        if not isinstance(entries, list):
            raise ValueError("native tool attachment entries are invalid")
        attachments = []
        for entry in entries:
            if (not isinstance(entry, Mapping) or set(entry) != {"path", "ref"} or not isinstance(entry["path"], list)
                    or not isinstance(entry["ref"], Mapping) or set(entry["ref"]) != {"store_id", "key", "digest", "size"}):
                raise ValueError("native tool attachment entry is invalid")
            attachments.append((tuple(entry["path"]), ObjectRef(**entry["ref"])))
        return cls(NativeMessageRoot.from_json(value["$native_tool_result"]), value["part_index"],
                   value["tool_name"], value["tool_call_id"], value["template"], value["digest"], value["size"],
                   value.get("attachment_format"), tuple(attachments))


@dataclass(frozen=True, slots=True)
class NativeToolArgumentsReference:
    """Project one published call into exact admitted canonical argument bytes."""

    root: NativeMessageRoot
    part_index: int
    tool_name: str
    tool_call_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native tool arguments requires one message")
        _count(self.part_index)
        if any(not isinstance(value, str) or not value for value in
               (self.tool_name, self.tool_call_id)):
            raise ValueError("native tool arguments identity is invalid")

    def to_json(self) -> dict[str, JsonValue]:
        return {"$native_tool_arguments": self.root.to_json(), "part_index": self.part_index,
                "tool_name": self.tool_name, "tool_call_id": self.tool_call_id}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeToolArgumentsReference":
        if not isinstance(value, Mapping) or set(value) != {
                "$native_tool_arguments", "part_index", "tool_name", "tool_call_id"}:
            raise ValueError("native tool arguments wire is invalid")
        return cls(NativeMessageRoot.from_json(value["$native_tool_arguments"]), value["part_index"],
                   value["tool_name"], value["tool_call_id"])


@dataclass(frozen=True, slots=True)
class NativeUserInputReference:
    """One immutable prompt part reconstructing a canonical text or text-list input."""
    root: NativeMessageRoot
    part_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.root, NativeMessageRoot) or self.root.count != 1:
            raise ValueError("native input reference requires one message")
        _count(self.part_index)

    def to_json(self) -> dict[str, JsonValue]:
        return {"$native_user_input": self.root.to_json(), "part_index": self.part_index}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeUserInputReference":
        if not isinstance(value, Mapping) or set(value) != {"$native_user_input", "part_index"}:
            raise ValueError("native user input wire is invalid")
        return cls(NativeMessageRoot.from_json(value["$native_user_input"]), value["part_index"])


__all__ = ["NativeToolArgumentsReference", "NativeUserInputReference", "NativeToolResultReference", "NativeTextReference", "NativeMessageRoot", "NativeMessageSequence", "NativeMessageSlice"]
