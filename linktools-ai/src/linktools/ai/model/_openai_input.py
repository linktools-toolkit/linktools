#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Encoding constraints of the OpenAI Chat adapter, independent of model support."""

from pydantic_ai.messages import BinaryContent, DocumentUrl


def openai_input_transport(media_type: str, input_kind: str, model: str) -> str | None:
    """Return the actual SDK wire transport, or None when it cannot encode it."""
    if input_kind == "uploaded":
        return None if media_type.startswith("image/") else "file-block"
    if input_kind == "image-url":
        return "image"
    if input_kind == "video-url":
        return None
    probe = BinaryContent(b"", media_type=media_type)
    if input_kind == "audio-url":
        return "audio" if _audio_supported(probe) else None
    if input_kind not in {"binary", "document-url"}:
        return None
    if _text_like(media_type):
        return "text-inline"
    if input_kind == "binary" and probe.is_image:
        return "image"
    if input_kind == "binary" and probe.is_audio:
        return "audio" if _audio_supported(probe) else None
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile(model)
    if profile is not None and not profile.get("openai_chat_supports_document_input", True):
        return None
    if input_kind == "document-url":
        try:
            DocumentUrl("https://input.invalid/file", media_type=media_type).format
        except (KeyError, ValueError):
            return None
        return "file-block"
    return "file-block" if probe.is_document else None


def _audio_supported(content: BinaryContent) -> bool:
    try:
        return content.is_audio and content.format in {"mp3", "wav"}
    except (KeyError, ValueError):
        return False


def _text_like(media_type: str) -> bool:
    # These MIME families are UTF-8 text blocks in the Chat adapter, not file parts.
    return (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml", "application/yaml", "application/x-yaml", "application/toml"}
        or media_type.endswith(("+json", "+xml"))
    )


__all__ = ["openai_input_transport"]
