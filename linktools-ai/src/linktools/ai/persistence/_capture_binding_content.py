#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent capture owners of the existing immutable Agent binding content."""

import hashlib
from collections.abc import Callable, Mapping

from ..agent import AgentBindingContract
from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ._archive import ContentArchive, PreparedArchive, archive_content_identity
from .codec import (
    agent_binding_content_summary, binding_reference, decode_agent_binding_content,
    decode_domain, encode_agent_binding_content, encode_domain,
)


BINDING_CODEC = "execution-binding"


def map_capture_binding_slots(
    key: str, value: JsonValue, transform: Callable[[Mapping[str, JsonValue]], JsonValue],
) -> JsonValue:
    """Visit declared contracts only; arbitrary input mappings remain opaque."""
    if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
        return value
    payload = dict(value["payload"])

    def contract(wire: JsonValue) -> JsonValue:
        if not isinstance(wire, Mapping) or wire.get("$dataclass") != "task_invocation_input_contract":
            return wire
        fields = wire.get("fields")
        if not isinstance(fields, Mapping) or "binding" not in fields:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        declaration = decode_domain(fields["binding"], dict[str, JsonValue])
        if declaration.get("type") != "agent":
            return wire
        config = declaration.get("config")
        if not isinstance(config, Mapping) or "binding_contract" not in config:
            return wire
        if not isinstance(config["binding_contract"], Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        declaration = {**declaration, "config": {**config, "binding_contract": transform(config["binding_contract"])}}
        return {**wire, "fields": {**fields, "binding": encode_domain(declaration)}}

    if key.startswith("v1/input-capture/agent/"):
        binding = payload.get("binding")
        if binding is not None:
            if not isinstance(binding, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload["binding"] = transform(binding)
        if payload.get("task_input") is not None:
            payload["task_input"] = contract(payload["task_input"])
    elif key.startswith("v1/input-capture/task/"):
        payload["contract"] = contract(payload.get("contract"))
    else:
        return value
    return {**value, "payload": payload}


def capture_binding_references(key: str, value: JsonValue) -> tuple[tuple[str, str, str], ...]:
    references = []
    def collect(binding: Mapping[str, JsonValue]) -> JsonValue:
        reference = binding_reference(binding)
        if reference is not None:
            references.append(reference)
        return binding
    map_capture_binding_slots(key, value, collect)
    return tuple(dict.fromkeys(references))


async def prepare_capture_bindings(
    key: str, value: JsonValue, archive: ContentArchive, prepared: PreparedArchive,
) -> JsonValue:
    bodies = {}
    def prepare(binding: Mapping[str, JsonValue]) -> JsonValue:
        if binding_reference(binding) is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        contract = AgentBindingContract.from_payload(binding)
        # Preserve additive fields and their exact JSON meaning on reconstruction.
        if canonical_json_bytes(contract.to_payload()) != canonical_json_bytes(binding):
            return binding
        raw = encode_agent_binding_content(contract)
        identity = archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest())
        bodies[identity] = raw
        return {"$execution_binding": identity, "archive_scope": archive.scope_id,
                "content_scope": archive.content_scope, "summary": agent_binding_content_summary(contract)}
    projected = map_capture_binding_slots(key, value, prepare)
    if bodies:
        await archive.prepare_content(tuple((BINDING_CODEC, raw) for raw in bodies.values()), prepared)
    return projected


def expand_capture_bindings(
    key: str, value: JsonValue, contents: Mapping[tuple[str, str, str], bytes],
) -> JsonValue:
    def expand(binding: Mapping[str, JsonValue]) -> JsonValue:
        reference = binding_reference(binding)
        if reference is None:
            return binding
        raw = contents.get(reference)
        if raw is None or archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest()) != reference[2]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        contract = decode_agent_binding_content(raw)
        if canonical_json_bytes(binding["summary"]) != canonical_json_bytes(agent_binding_content_summary(contract)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return contract.to_payload()
    return map_capture_binding_slots(key, value, expand)


async def restore_capture_bindings(key: str, value: JsonValue, archive: ContentArchive) -> JsonValue:
    references = capture_binding_references(key, value)
    if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raws = await archive.read_contents(tuple(reference[2] for reference in references), codec=BINDING_CODEC) if references else ()
    return expand_capture_bindings(key, value, dict(zip(references, raws, strict=True)))


__all__ = ["map_capture_binding_slots", "capture_binding_references", "prepare_capture_bindings",
           "expand_capture_bindings", "restore_capture_bindings"]
