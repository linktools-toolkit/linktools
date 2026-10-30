#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-level durable codec for Pydantic user-content items."""

import base64
from collections.abc import Callable, Mapping
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
from . import ReferencedContent
from ._wire_fields import decode_fields


class RetiredReferencedContent(TextContent):
    """Keep recovery provenance outside SDK-visible dataclass fields."""

    def __init__(self, reference: ReferencedContent, summary: str) -> None:
        super().__init__(summary)
        self.reference = reference

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.content == other.content and self.reference == other.reference


def encode_user_content_item(item: UserContent) -> JsonValue:
    try:
        if isinstance(item, str):
            return {"kind": "text", "text": item}
        if isinstance(item, RetiredReferencedContent):
            return {"kind": "text-content", "content": item.content,
                    "metadata": {"referenced_content": encode_user_content_item(item.reference)}}
        if isinstance(item, ReferencedContent):
            return {"kind": "referenced-content", "ref": item.ref.to_payload(),
                    "source": {"kind": item.source.kind, "id": item.source.id,
                               "revision": item.source.revision},
                    "media_type": item.media_type,
                    **({"identifier": item.identifier} if item.identifier is not None else {}),
                    **({"representation": item.representation} if item.representation != "binary" else {}),
                    **({"source_parameters": dict(item.source_parameters)} if item.source_parameters else {})}
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
    if kind == "referenced-content":
        from ..asset import AssetVersionRef
        from ..spec import CapabilityRef
        source = value.get("source")
        required = {"kind", "ref", "media_type", "source"}
        if (not required.issubset(value) or set(value) - required - {"representation", "source_parameters", "identifier"}
            or not isinstance(value.get("source_parameters", {}), Mapping)
            or not isinstance(source, Mapping) or set(source) != {"kind", "id", "revision"}):
            raise ValueError("referenced content is invalid")
        return ReferencedContent(AssetVersionRef.from_payload(value["ref"]),
                                 media_type=value["media_type"],
                                 representation=value.get("representation", "binary"),
                                 source_parameters=value.get("source_parameters", {}),
                                 identifier=_optional_string(value.get("identifier")),
                                 source=CapabilityRef(source["kind"], source["id"], source["revision"]))
    if kind == "text":
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("text content is invalid")
        return text
    if kind == "text-content":
        kwargs = _content_fields(value, TextContent, {
            "content": _required_string,
            "metadata": lambda raw: raw,
        })
        metadata = kwargs.get("metadata")
        if isinstance(metadata, Mapping) and set(metadata) == {"referenced_content"}:
            reference = decode_user_content_item(metadata["referenced_content"])
            if not isinstance(reference, ReferencedContent):
                raise ValueError("retired reference descriptor is invalid")
            return RetiredReferencedContent(reference, kwargs["content"])
        return TextContent(**kwargs)
    if kind == "binary":
        return BinaryContent(**_content_fields(value, BinaryContent, {
            "data": lambda raw: base64.b64decode(_required_string(raw), validate=True),
            "media_type": _media_type,
            "identifier": _optional_string,
            "vendor_metadata": lambda raw: raw,
        }))
    url_types = {
        "image-url": ImageUrl,
        "audio-url": AudioUrl,
        "document-url": DocumentUrl,
        "video-url": VideoUrl,
    }
    url_type = url_types.get(kind)
    if url_type is not None:
        return url_type(**_content_fields(value, url_type, {
            "url": _required_string,
            "media_type": _optional_string,
            "force_download": _force_download,
            "identifier": _optional_string,
            "vendor_metadata": lambda raw: raw,
        }))
    if kind == "uploaded-file":
        return UploadedFile(**_content_fields(value, UploadedFile, {
            "file_id": _required_string,
            "provider_name": _required_string,
            "media_type": _optional_string,
            "identifier": _optional_string,
            "vendor_metadata": lambda raw: raw,
        }))
    if kind == "cache-point":
        return CachePoint(**_content_fields(value, CachePoint, {"ttl": _cache_ttl}))
    raise UnsupportedUserContentKind("unknown user content kind")


def _content_fields(
    value: Mapping[str, object],
    target: type,
    decoders: Mapping[str, Callable[[object], object]],
) -> dict[str, object]:
    # Ordinary content historically tolerates extra keys, unlike reference descriptors.
    selected = {key: raw for key, raw in value.items() if key == "kind" or key in decoders}
    return decode_fields(selected, target, decoders, required={"kind"})


def _required_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("user content string is invalid")
    return value


def _media_type(value: object) -> str:
    result = _required_string(value)
    if not result:
        raise ValueError("binary content media type is invalid")
    return result


def _force_download(value: object) -> object:
    if value not in {False, True, "allow-local"}:
        raise ValueError("URL download mode is invalid")
    return value


def _cache_ttl(value: object) -> str:
    if value not in {"5m", "1h"}:
        raise ValueError("cache point is invalid")
    return cast(str, value)


class UnsupportedUserContentKind(ValueError):
    pass


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("user content identifier is invalid")
    return cast(str | None, value)


__all__: list[str] = []
