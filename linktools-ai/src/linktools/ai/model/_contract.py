#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical model binding and resolution contracts."""

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic_ai.messages import UploadedFile, UploadedFileProviderName
from pydantic_ai.models import Model

from ..core import JsonValue


class ModelBinding(Protocol):
    @property
    def route_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def vision(self) -> bool: ...

    @property
    def semantic_payload(self) -> Mapping[str, JsonValue]: ...

    @property
    def fingerprint(self) -> str: ...

    def materialize(self) -> Model:
        """Build the provider model; expected configuration failures raise AIError."""
        ...


class ModelResolver(Protocol):
    def resolve(self, route_id: str) -> ModelBinding: ...

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: "str | None" = None,
    ) -> ModelBinding: ...


class LinkToolsUploadedFile(UploadedFile):
    """UploadedFile with LinkTools-owned media type provenance."""

    declared_media_type: str | None

    def __init__(
        self,
        file_id: str,
        provider_name: UploadedFileProviderName,
        *,
        media_type: str | None = None,
        vendor_metadata: dict[str, Any] | None = None,
        identifier: str | None = None,
        kind: Literal["uploaded-file"] = "uploaded-file",
    ) -> None:
        super().__init__(
            file_id,
            provider_name,
            media_type=media_type,
            vendor_metadata=vendor_metadata,
            identifier=identifier,
            kind=kind,
        )
        object.__setattr__(self, "declared_media_type", media_type)


def declared_uploaded_file_media_type(value: UploadedFile) -> str | None:
    """Return only a media type explicitly supplied to UploadedFile."""
    if not isinstance(value, UploadedFile):
        raise TypeError("value must be UploadedFile")
    if not isinstance(value, LinkToolsUploadedFile):
        return None
    return value.declared_media_type


__all__ = [
    "LinkToolsUploadedFile",
    "ModelBinding",
    "ModelResolver",
    "declared_uploaded_file_media_type",
]
