#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact accepted argument reconstruction from fixed published call parts."""

from pydantic_ai.messages import ModelResponse, ToolCallPart

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..model.messages import decode_model_messages
from ..storage import StoredPayload
from ._message_root import NativeMessageRoot, NativeToolArgumentsReference


def _source(raw: bytes, part_index: int) -> ToolCallPart:
    messages = decode_model_messages(raw)
    if (len(messages) != 1 or not isinstance(messages[0], ModelResponse)
            or not 0 <= part_index < len(messages[0].parts)
            or not isinstance(messages[0].parts[part_index], ToolCallPart)):
        raise ValueError("invalid tool argument source")
    return messages[0].parts[part_index]


def restore_native_tool_arguments_payload(reference: NativeToolArgumentsReference, raw: bytes) -> StoredPayload:
    """Reconstruct and validate canonical bytes without applying tool defaults."""
    try:
        source = _source(raw, reference.part_index)
        if (source.tool_name != reference.tool_name or source.tool_call_id != reference.tool_call_id):
            raise ValueError("tool argument identity mismatch")
        arguments = source.args_as_dict()
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a mapping")
        payload = StoredPayload.inline_bytes(canonical_json_bytes(arguments))
        if (payload.digest, payload.size) != (reference.digest, reference.size):
            raise ValueError("tool argument checksum mismatch")
        return payload
    except (TypeError, ValueError, IndexError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def prepare_native_tool_arguments(
    root: NativeMessageRoot, raw: bytes, part_index: int, payload: StoredPayload,
) -> NativeToolArgumentsReference | None:
    """Select only a complete byte-equivalent accepted argument payload."""
    if payload.kind != "inline" or payload.encoding != "base64":
        return None
    source = _source(raw, part_index)
    reference = NativeToolArgumentsReference(root, part_index, source.tool_name, source.tool_call_id,
                                            payload.digest, payload.size)
    try:
        restored = restore_native_tool_arguments_payload(reference, raw)
    except AIError:
        # Validated defaults/coercions and unsupported model argument forms
        # remain owned by the admitted canonical payload, never approximated.
        return None
    return reference if restored == payload else None


__all__ = ["prepare_native_tool_arguments", "restore_native_tool_arguments_payload"]
