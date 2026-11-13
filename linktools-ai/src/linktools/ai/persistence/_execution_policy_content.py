#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed Runtime policy owners and explicit per-execution changes."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..execution import ExecutionRecord
from ._archive import archive_content_identity
from .codec import (EXECUTION_POLICY_FIELDS, decode_envelope, encode_envelope,
                    encode_domain, runtime_policy_reference)
from .state import StoredRecord

POLICY_CODEC = "runtime-policy"


def policy_reference(record: StoredRecord) -> tuple[str, str, str] | None:
    return runtime_policy_reference(decode_envelope(record.data).value["payload"])


def policy_values(value: ExecutionRecord) -> dict[str, JsonValue]:
    return {name: encode_domain(getattr(value, name)) for name in EXECUTION_POLICY_FIELDS}


def encode_policy(values: Mapping[str, JsonValue]) -> tuple[str, bytes]:
    raw = canonical_json_bytes({"version": 1, "fields": dict(values)})
    return archive_content_identity(POLICY_CODEC, hashlib.sha256(raw).hexdigest()), raw


def decode_policy(raw: bytes) -> Mapping[str, JsonValue]:
    try:
        value = json.loads(raw)
        if (not isinstance(value, Mapping) or set(value) != {"version", "fields"}
                or type(value["version"]) is not int or value["version"] != 1
                or not isinstance(value["fields"], Mapping)
                or set(value["fields"]) != set(EXECUTION_POLICY_FIELDS)):
            raise ValueError("invalid Runtime policy owner")
        return value["fields"]
    except (UnicodeError, ValueError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def expand_policy(record: StoredRecord, raw: bytes) -> StoredRecord:
    reference = policy_reference(record)
    if reference is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return expand_policy_records((record,), {reference: raw})[0]


def expand_policy_records(records: Sequence[StoredRecord], contents: Mapping[tuple[str, str, str], bytes]) -> tuple[StoredRecord, ...]:
    """Decode each immutable policy once, then apply each execution's own overrides."""
    decoded = {}
    expanded = []
    for record in records:
        reference = policy_reference(record)
        if reference is None:
            expanded.append(record)
            continue
        if reference not in decoded:
            decoded[reference] = decode_policy(contents[reference])
        values = decoded[reference]
        envelope = decode_envelope(record.data)
        payload = envelope.value["payload"]
        fields = dict(payload["fields"])
        overrides = fields.pop("runtime_overrides", {})
        if (not isinstance(overrides, Mapping) or set(overrides) - set(EXECUTION_POLICY_FIELDS)
                or set(fields).intersection(EXECUTION_POLICY_FIELDS)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fields.pop("runtime_policy")
        fields.update(values)
        fields.update(overrides)
        expanded.append(replace(record, data=encode_envelope({**envelope.value,
            "payload": {**payload, "fields": fields}})))
    return tuple(expanded)


__all__ = ["POLICY_CODEC", "policy_reference", "policy_values", "encode_policy", "decode_policy", "expand_policy", "expand_policy_records"]
