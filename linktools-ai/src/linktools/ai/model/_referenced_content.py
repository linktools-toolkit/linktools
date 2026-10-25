#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descriptor-only input whose immutable bytes remain with their Asset owner."""

import json
from collections.abc import Mapping
from typing import Literal, TYPE_CHECKING

from pydantic_ai.messages import TextContent

from ..asset import AssetVersionRef
from ..core import ImmutableJsonMapping
from ..spec import CapabilityRef


if TYPE_CHECKING:
    from ..core import JsonValue


class ReferencedContent(TextContent):
    """Deliver a recoverable Asset without copying its bytes into Runtime storage.

    The selected Agent's ReferencedInputsCapability authorizes and loads this
    exact version for each provider request. The host must retain that version;
    a URL alone, or a mutable path resolved to its latest value, is insufficient.
    Runtime-owned codecs preserve this type; SDK JSON is not its durable codec.
    Descriptor attributes are not dataclass fields: SDK serialization exposes only
    the inherited text summary, including before provider hydration.
    Binary representation keeps file-input admission and encoding. Explicit text
    representation decodes verified UTF-8 bytes into an ordinary provider string;
    MIME metadata alone never chooses or changes the representation.
    Optional source_parameters carry immutable JSON reconstruction arguments for
    the selected loader, not content bodies, credentials or authorization proof.
    They are retained with the descriptor but omitted from its model-visible summary.
    An optional identifier associates an input occurrence with Runtime attachment
    facts. It is not Asset identity, a reconstruction argument or authorization.
    """

    ref: AssetVersionRef
    media_type: str
    source: CapabilityRef
    representation: "Literal['binary', 'text']"
    identifier: str | None
    source_parameters: Mapping[str, object]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (self.ref, self.media_type, self.source, self.representation,
                self.identifier, self.source_parameters, self.content, self.metadata) == (
                    other.ref, other.media_type, other.source, other.representation,
                    other.identifier, other.source_parameters, other.content, other.metadata)

    def __init__(
        self, ref: AssetVersionRef, *, media_type: str, source: CapabilityRef,
        representation: "Literal['binary', 'text']" = "binary",
        source_parameters: "Mapping[str, JsonValue] | None" = None,
        identifier: str | None = None,
    ) -> None:
        if not isinstance(ref, AssetVersionRef):
            raise TypeError("referenced content requires an AssetVersionRef")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("referenced content media_type must be non-empty")
        if (not isinstance(source, CapabilityRef) or source.kind != "capability"
            or not isinstance(source.id, str) or not source.id.strip()
            or type(source.revision) is not int or source.revision < 1):
            raise ValueError("referenced content source must name a capability revision")
        if representation not in ("binary", "text"):
            raise ValueError("referenced content representation must be binary or text")
        if source_parameters is not None and not isinstance(source_parameters, Mapping):
            raise TypeError("referenced content source_parameters must be a JSON mapping")
        if identifier is not None and not isinstance(identifier, str):
            raise TypeError("referenced content identifier must be a string")
        self.source_parameters = ImmutableJsonMapping({} if source_parameters is None else source_parameters)
        self.ref = ref
        self.media_type = media_type
        self.source = source
        self.representation = representation
        self.identifier = identifier
        super().__init__(json.dumps(
            {"attachment": "reference", "size": ref.size, "mime_type": media_type,
             **({"representation": representation} if representation != "binary" else {})},
            ensure_ascii=True, separators=(",", ":"),
        ))


__all__ = ["ReferencedContent"]
