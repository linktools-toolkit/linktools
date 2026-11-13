#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable native owner rows, with occurrence order and replay identities intact."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import base64
import binascii
import json

from ..steps import encode_native_store_key, decode_native_store_key, decode_native_replay_key
from ._message_root import NativeMessageRoot, NativeMessageSequence, NativeMessageSlice


@dataclass(frozen=True, slots=True)
class NativeStepBackup:
    """Raw owner inventory; message bodies belong to the shared message closure.

    Row position is insertion order. Surrogate SQL IDs and audit timestamps are
    not identities; keyed replays use their original run/key pair, while unkeyed
    replay rows retain every occurrence in insertion order.
    """

    store_key: bytes = field(repr=False)
    runs: tuple[Mapping[str, object], ...] = field(repr=False)
    events: tuple[Mapping[str, object], ...] = field(repr=False)
    snapshots: tuple[Mapping[str, object], ...] = field(repr=False)

    def __post_init__(self) -> None:
        NativeMessageRoot("validation", self.store_key)
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
        return tuple(NativeMessageRoot(group_id, self.store_key, value.slices)
                     for value in self.message_sequences)

    def to_json(self) -> dict[str, object]:
        result = {"type": "native_step_backup", "version": 1,
                  "store_key": encode_native_store_key(self.store_key)}
        for name in ("runs", "events", "snapshots"):
            rows = []
            for row in getattr(self, name):
                value = deepcopy(dict(row))
                value["store_key"] = encode_native_store_key(value["store_key"])
                for key in ("idempotency_key", "managed_start_key"):
                    if key not in value or (key == "managed_start_key" and value[key] is None):
                        continue
                    replay = decode_native_replay_key(value[key])
                    if key == "managed_start_key" and replay is None:
                        raise ValueError("managed start identity must be a public replay key")
                    value[key] = base64.b64encode(value[key]).decode("ascii")
                rows.append(value)
            result[name] = rows
        return result

    @classmethod
    def from_json(cls, value: object) -> "NativeStepBackup":
        if (not isinstance(value, Mapping) or set(value) != {
            "type", "version", "store_key", "runs", "events", "snapshots",
        } or value["type"] != "native_step_backup" or type(value["version"]) is not int
                or value["version"] != 1
                or any(not isinstance(value[name], list) for name in ("runs", "events", "snapshots"))):
            raise ValueError("native backup wire is invalid")
        batches = []
        for name in ("runs", "events", "snapshots"):
            rows = []
            for row in value[name]:
                if not isinstance(row, Mapping):
                    raise ValueError("native backup row is invalid")
                decoded = dict(row)
                decoded["store_key"] = decode_native_store_key(decoded.get("store_key"))
                for key in ("idempotency_key", "managed_start_key"):
                    if key not in decoded or (key == "managed_start_key" and decoded[key] is None):
                        continue
                    token = decoded[key]
                    try:
                        raw = base64.b64decode(token, validate=True)
                        if base64.b64encode(raw).decode("ascii") != token:
                            raise ValueError("native replay transport is invalid")
                        replay = decode_native_replay_key(raw)
                        if key == "managed_start_key" and replay is None:
                            raise ValueError("managed start identity must be a public replay key")
                    except (TypeError, binascii.Error) as error:
                        raise ValueError("native replay transport is invalid") from error
                    decoded[key] = raw
                rows.append(decoded)
            batches.append(tuple(rows))
        return cls(decode_native_store_key(value["store_key"]), *batches)


__all__ = ["NativeStepBackup"]
