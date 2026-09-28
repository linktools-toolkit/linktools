#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-level durable codec for Pydantic user-content items."""

import base64
from collections.abc import Mapping
from typing import cast

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UserContent,
    UploadedFile,
    VideoUrl,
)

from ..core import JsonValue, normalize_json_value
from ..errors import AIError, ErrorCode


def encode_user_content_item(item: UserContent) -> JsonValue:
    try:
        if isinstance(item, str):
            return {"kind": "text", "text": item}
        if isinstance(item, TextContent):
            return {
                "kind": "text-content",
                "content": item.content,
                "metadata": normalize_json_value(item.metadata),
            }
        if isinstance(item, BinaryContent):
            return {
                "kind": "binary",
                "data": base64.b64encode(item.data).decode("ascii"),
                "media_type": item.media_type,
                "identifier": item.identifier,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
            return {
                "kind": item.kind,
                "url": item.url,
                "media_type": item.media_type,
                "identifier": item.identifier,
                "force_download": item.force_download,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, UploadedFile):
            return {
                "kind": "uploaded-file",
                "file_id": item.file_id,
                "provider_name": item.provider_name,
                "media_type": item.media_type,
                "identifier": item.identifier,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, CachePoint):
            return {"kind": "cache-point", "ttl": item.ttl}
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def decode_user_content_item(value: object) -> UserContent:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise ValueError("user content item is invalid")
    kind = value["kind"]
    if kind == "text":
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("text content is invalid")
        return text
    if kind == "text-content":
        content = value.get("content")
        if not isinstance(content, str) or "metadata" not in value:
            raise ValueError("text content is invalid")
        return TextContent(content, metadata=value["metadata"])
    if kind == "binary":
        data = value.get("data")
        media_type = value.get("media_type")
        if (
            not isinstance(data, str)
            or not isinstance(media_type, str)
            or not media_type
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("binary content is invalid")
        return BinaryContent(
            base64.b64decode(data, validate=True),
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            vendor_metadata=value["vendor_metadata"],
        )
    url_types = {
        "image-url": ImageUrl,
        "audio-url": AudioUrl,
        "document-url": DocumentUrl,
        "video-url": VideoUrl,
    }
    url_type = url_types.get(kind)
    if url_type is not None:
        url = value.get("url")
        media_type = value.get("media_type")
        force_download = value.get("force_download")
        if (
            not isinstance(url, str)
            or media_type is not None
            and not isinstance(media_type, str)
            or force_download not in {False, True, "allow-local"}
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("URL content is invalid")
        return url_type(
            url,
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            force_download=force_download,
            vendor_metadata=value["vendor_metadata"],
        )
    if kind == "uploaded-file":
        file_id = value.get("file_id")
        provider_name = value.get("provider_name")
        media_type = value.get("media_type")
        if (
            not isinstance(file_id, str)
            or not isinstance(provider_name, str)
            or media_type is not None
            and not isinstance(media_type, str)
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("uploaded file is invalid")
        return UploadedFile(
            file_id,
            provider_name,
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            vendor_metadata=value["vendor_metadata"],
        )
    if kind == "cache-point":
        ttl = value.get("ttl")
        if ttl not in {"5m", "1h"}:
            raise ValueError("cache point is invalid")
        return CachePoint(ttl=cast(str, ttl))
    raise UnsupportedUserContentKind("unknown user content kind")


class UnsupportedUserContentKind(ValueError):
    pass


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("user content identifier is invalid")
    return cast(str | None, value)


__all__: list[str] = []
