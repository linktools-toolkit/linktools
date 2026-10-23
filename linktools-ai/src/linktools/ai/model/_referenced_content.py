#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descriptor-only input whose immutable bytes remain with their Asset owner."""

import json
from dataclasses import dataclass

from pydantic_ai.messages import TextContent

from ..asset import AssetVersionRef
from ..spec import CapabilityRef


@dataclass(init=False)
class ReferencedContent(TextContent):
    """Deliver a recoverable Asset without copying its bytes into Runtime storage.

    The selected Agent's ReferencedInputsCapability authorizes and loads this
    exact version for each provider request. The host must retain that version;
    a URL alone, or a mutable path resolved to its latest value, is insufficient.
    Runtime-owned codecs preserve this type; SDK JSON is not its durable codec.
    """

    ref: AssetVersionRef
    media_type: str
    source: CapabilityRef

    def __init__(self, ref: AssetVersionRef, *, media_type: str, source: CapabilityRef) -> None:
        if not isinstance(ref, AssetVersionRef):
            raise TypeError("referenced content requires an AssetVersionRef")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("referenced content media_type must be non-empty")
        if (not isinstance(source, CapabilityRef) or source.kind != "capability"
            or not isinstance(source.id, str) or not source.id.strip()
            or type(source.revision) is not int or source.revision < 1):
            raise ValueError("referenced content source must name a capability revision")
        self.ref = ref
        self.media_type = media_type
        self.source = source
        super().__init__(json.dumps(
            {"attachment": "reference", "size": ref.size, "mime_type": media_type},
            ensure_ascii=True, separators=(",", ":"),
        ))


__all__ = ["ReferencedContent"]
