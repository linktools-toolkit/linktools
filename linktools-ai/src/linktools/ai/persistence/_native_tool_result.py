#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact operation receipt reconstruction from immutable published tool returns."""

from dataclasses import replace
import json

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from ..core import canonical_json_bytes
from ..model.attachments import restore_attachment_payload, validate_attachment_payload
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
        reconstructed = encode_model_messages((replace(templates[0], parts=[replace(original, content=source.content)]),))
        result = (StoredPayload.inline_bytes(reconstructed) if reference.attachment_format is None else
                  restore_attachment_payload(reconstructed, format=reference.attachment_format,
                                             attachments=reference.attachment_entries()))
        if result.digest != reference.digest or result.size != reference.size:
            raise ValueError("native tool receipt checksum mismatch")
        return result
    except (TypeError, ValueError, IndexError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def prepare_native_tool_result(
    root: NativeMessageRoot, raw: bytes, part_index: int, payload: StoredPayload, *, operation_id: str,
) -> NativeToolResultReference | None:
    """Select canonical receipts whose complete bytes reconstruct from one part."""
    if payload.kind != "inline" or payload.encoding not in {"base64", "json-attachments"}:
        return None
    attachments = ()
    attachment_format = None
    try:
        if payload.encoding == "json-attachments":
            manifest = payload.value
            if manifest["format"] != "model-messages":
                return None
            original = manifest["content"]
            source = json.loads(raw)
            if (not isinstance(original, list) or len(original) != 1 or len(original[0]["parts"]) != 1
                    or not isinstance(source, list) or len(source) != 1):
                return None
            original[0]["parts"][0]["content"] = source[0]["parts"][part_index]["content"]
            value = canonical_json_bytes(original)
            validate_attachment_payload(value, payload)
            attachment_format = manifest["format"]
            attachments = tuple((tuple(entry["path"]), ref) for entry, ref in
                                zip(manifest["attachments"], payload.object_refs(), strict=True))
        else:
            value = payload.decode()
        if not isinstance(value, bytes):
            return None
        messages = decode_model_messages(value)
    except (AIError, TypeError, ValueError, KeyError, IndexError):
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
                                         template, payload.digest, payload.size, attachment_format, attachments)
    restored = restore_native_tool_result_payload(reference, raw)
    return reference if canonical_json_bytes(restored.to_json()) == canonical_json_bytes(payload.to_json()) else None


__all__ = ["prepare_native_tool_result", "restore_native_tool_result_payload"]
