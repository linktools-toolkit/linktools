#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable native owner rows, with occurrence order and replay identities intact."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json

from ._message_root import NativeMessageRoot, NativeMessageSequence, NativeMessageSlice


@dataclass(frozen=True, slots=True)
class NativeStepBackup:
    """Raw owner inventory; message bodies belong to the shared message closure.

    Row position is insertion order. Surrogate SQL IDs and audit timestamps are
    not identities; replay keys, including unkeyed occurrence digests, are.
    """

    store_digest: str
    runs: tuple[Mapping[str, object], ...]
    events: tuple[Mapping[str, object], ...]
    snapshots: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        NativeMessageRoot("validation", self.store_digest)
        for name in ("runs", "events", "snapshots"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(row, Mapping) for row in values):
                raise ValueError("native backup rows must be ordered mappings")
            object.__setattr__(self, name, tuple(deepcopy(dict(row)) for row in values))

    @property
    def message_sequences(self) -> tuple[NativeMessageSequence, ...]:
        result = []
        for row in self.snapshots:
            if row.get("pruned") is True:
                continue
            descriptor = json.loads(row["messages"])
            count = descriptor["stored_messages"]
            if type(count) is not int or count < 0:
                raise ValueError("native backup message count is invalid")
            result.append(NativeMessageSequence(() if count == 0 else (
                NativeMessageSlice(row["sequence_root"], count, 0, count),
            )))
        return tuple(result)

    def message_roots(self, group_id: str) -> tuple[NativeMessageRoot, ...]:
        return tuple(NativeMessageRoot(group_id, self.store_digest, value.slices)
                     for value in self.message_sequences)

    def to_json(self) -> dict[str, object]:
        return {"type": "native_step_backup", "version": 1, "store_digest": self.store_digest,
                **{name: [deepcopy(dict(row)) for row in getattr(self, name)]
                   for name in ("runs", "events", "snapshots")}}

    @classmethod
    def from_json(cls, value: object) -> "NativeStepBackup":
        if (not isinstance(value, Mapping) or set(value) != {
            "type", "version", "store_digest", "runs", "events", "snapshots",
        } or value["type"] != "native_step_backup" or type(value["version"]) is not int
                or value["version"] != 1
                or any(not isinstance(value[name], list) for name in ("runs", "events", "snapshots"))):
            raise ValueError("native backup wire is invalid")
        return cls(value["store_digest"], *(tuple(value[name]) for name in ("runs", "events", "snapshots")))


__all__ = ["NativeStepBackup"]
