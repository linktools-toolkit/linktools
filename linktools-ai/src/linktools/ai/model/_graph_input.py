#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Graph input capture and precise attachment provenance."""

from collections.abc import Mapping, Sequence

from pydantic_ai.messages import BinaryContent, UserContent

from ..core import JsonValue, normalize_json_value
from ..errors import AIError
from ..storage import StoredPayload
from ._user_input import StoredUserInput, capture_plain_text_input, plain_text_input_content


_GRAPH_ATTACHMENTS = "linktools.graph_attachments"


def native_graph_input_metadata(value: StoredUserInput) -> dict[str, JsonValue] | None:
    """Carry only accepted file provenance that the native message cannot derive."""
    if value.view is None:
        return None
    files, prompt = value.view.get("files"), value.view.get("prompt")
    if (not isinstance(files, list) or not isinstance(prompt, Mapping)
            or prompt.get("kind") not in {"text", "items"}
            or any(not isinstance(item, Mapping) or not isinstance(item.get("path"), str) for item in files)):
        return None
    items = prompt.get("items")
    if not files and (not isinstance(items, list) or not any(
        isinstance(item, Mapping) and item.get("kind") == "binary" for item in items
    )):
        return None
    return {_GRAPH_ATTACHMENTS: {"paths": [item["path"] for item in files], "scalar_prompt": prompt["kind"] == "text"}}


def capture_native_graph_input(
    content: str | Sequence[UserContent], metadata: Mapping[str, JsonValue] | None = None,
) -> StoredUserInput:
    """Reconstruct accepted direct binaries and independent files in materializer order."""
    if metadata is None:
        return capture_plain_text_input(content)
    import hashlib

    from ..workspace import normalize_workspace_input_path
    from ._attachment import binary_attachment_text, input_attachment_views
    from ._user_content_codec import encode_user_content_item

    descriptor = metadata.get(_GRAPH_ATTACHMENTS)
    if (set(metadata) != {_GRAPH_ATTACHMENTS} or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"paths", "scalar_prompt"} or type(descriptor["scalar_prompt"]) is not bool
            or not isinstance(descriptor["paths"], list) or isinstance(content, str)):
        raise ValueError("invalid Graph attachment provenance")
    paths = descriptor["paths"]
    offset = 2 * len(paths)
    if len(content) <= offset:
        raise ValueError("Graph attachments lack their original prompt")

    files: list[dict[str, JsonValue]] = []
    for index, path in enumerate(paths):
        label, binary = content[2 * index:2 * index + 2]
        if (not isinstance(path, str) or normalize_workspace_input_path(path) != path
                or type(binary) is not BinaryContent or label != binary_attachment_text(binary, path=path)):
            raise ValueError("Graph file differs from its admitted path")
        files.append({"path": path, "media_type": binary.media_type, "size": len(binary.data),
                      "digest": hashlib.sha256(binary.data).hexdigest()})

    original_items: list[UserContent] = []
    items: list[JsonValue] = []
    while offset < len(content):
        item = content[offset]
        offset += 1
        if isinstance(item, str):
            items.append({"kind": "text", "text": item})
        elif offset < len(content) and type(content[offset]) is BinaryContent:
            binary = content[offset]
            if item != binary_attachment_text(binary):
                raise ValueError("Graph binary lacks its original materializer framing")
            item = binary
            offset += 1
            items.append({
                "kind": "binary", "digest": hashlib.sha256(binary.data).hexdigest(), "size": len(binary.data),
                "media_type": binary.media_type, "identifier": binary.identifier,
                "vendor_metadata": normalize_json_value(binary.vendor_metadata),
            })
        else:
            raise ValueError("unsupported Graph attachment occurrence")
        original_items.append(item)
    if not files and not any(type(item) is BinaryContent for item in original_items):
        raise ValueError("Graph attachment provenance has no attachment")
    if descriptor["scalar_prompt"]:
        if len(original_items) != 1 or not isinstance(original_items[0], str):
            raise ValueError("Graph scalar prompt differs from its original content")
        original = original_items[0]
        prompt = items[0]
    else:
        original = tuple(original_items)
        prompt = {"kind": "items", "items": items}
    attachments = input_attachment_views(original, files)
    if files:
        # Match the materializer's presentation order without changing occurrence IDs.
        attachments = (*attachments[-len(files):], *attachments[:-len(files)])
    return StoredUserInput(
        "user-content-v1", StoredPayload.inline_json({"items": [encode_user_content_item(item) for item in content]}),
        {"version": 1, "prompt": prompt, "files": files, "attachments": list(attachments)},
    )


def native_graph_input_content(value: StoredUserInput) -> str | list[UserContent] | None:
    """Select only a byte-exact, already frozen turn prompt."""
    from ._user_content_codec import decode_user_content_item

    plain = plain_text_input_content(value)
    if plain is not None:
        return plain
    metadata = native_graph_input_metadata(value)
    if metadata is None or value.codec != "user-content-v1" or value.payload.kind != "inline" or value.payload.encoding != "json":
        return None
    try:
        decoded = value.payload.decode()
        if not isinstance(decoded, Mapping) or set(decoded) != {"items"} or not isinstance(decoded["items"], list):
            return None
        content = [decode_user_content_item(item) for item in decoded["items"]]
        return content if capture_native_graph_input(content, metadata) == value else None
    except (TypeError, ValueError, AIError):
        return None
