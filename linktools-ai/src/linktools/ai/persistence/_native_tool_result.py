#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact operation receipt reconstruction from immutable published tool returns."""

from dataclasses import replace

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from ..errors import AIError, ErrorCode
from ..model.messages import decode_model_messages, encode_model_messages
from ..storage import StoredPayload
from ._message_root import NativeMessageRoot, NativeToolResultReference


def restore_native_tool_result_payload(reference: NativeToolResultReference, raw: bytes) -> StoredPayload:
    """Verify source identity and reconstruct the original receipt byte for byte."""
    try:
        messages = decode_model_messages(raw)
        templates = decode_model_messages(reference.template.encode("utf-8"))
        if (len(messages) != 1 or not isinstance(messages[0], ModelRequest)
                or reference.part_index >= len(messages[0].parts)
                or len(templates) != 1 or not isinstance(templates[0], ModelRequest)
                or len(templates[0].parts) != 1):
            raise ValueError("invalid tool result source")
        source, original = messages[0].parts[reference.part_index], templates[0].parts[0]
        if (not isinstance(source, ToolReturnPart) or not isinstance(original, ToolReturnPart)
                or original.content is not None
                or original.tool_name != "runtime"
                or source.tool_name != reference.tool_name or source.tool_call_id != reference.tool_call_id):
            raise ValueError("invalid tool return projection")
        result = StoredPayload.inline_bytes(encode_model_messages((
            replace(templates[0], parts=[replace(original, content=source.content)]),)))
        if result.digest != reference.digest or result.size != reference.size:
            raise ValueError("native tool receipt checksum mismatch")
        return result
    except (TypeError, ValueError, IndexError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def prepare_native_tool_result(
    root: NativeMessageRoot, raw: bytes, part_index: int, payload: StoredPayload, *, operation_id: str,
) -> NativeToolResultReference | None:
    """Select canonical receipts whose complete bytes reconstruct from one part."""
    if payload.kind != "inline" or payload.encoding != "base64":
        return None
    value = payload.decode()
    if not isinstance(value, bytes):
        return None
    try:
        messages = decode_model_messages(value)
    except (AIError, TypeError, ValueError):
        return None
    sources = decode_model_messages(raw)
    if (len(messages) != 1 or not isinstance(messages[0], ModelRequest) or len(messages[0].parts) != 1
            or len(sources) != 1 or not isinstance(sources[0], ModelRequest)
            or not 0 <= part_index < len(sources[0].parts)):
        return None
    original, source = messages[0].parts[0], sources[0].parts[part_index]
    if (not isinstance(original, ToolReturnPart) or not isinstance(source, ToolReturnPart)
            or original.tool_name != "runtime" or original.tool_call_id != operation_id):
        return None
    # Selection must not change a valid noncanonical receipt into a fatal
    # publication error. Strict restoration remains the durable read contract.
    if encode_model_messages(messages) != value:
        return None
    reconstructed = encode_model_messages((
        replace(messages[0], parts=[replace(original, content=source.content)]),))
    if reconstructed != value:
        return None
    template = encode_model_messages((replace(messages[0], parts=[replace(original, content=None)]),)).decode("utf-8")
    reference = NativeToolResultReference(root, part_index, source.tool_name, source.tool_call_id,
                                         template, payload.digest, payload.size)
    restored = restore_native_tool_result_payload(reference, raw)
    return reference if restored.decode() == value else None


__all__ = ["prepare_native_tool_result", "restore_native_tool_result_payload"]
