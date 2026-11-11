#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original model client bytes in the existing interaction payload lifecycle."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, cast

from ..errors import AIError, ErrorCode
from ..model import ModelClientRequest, ModelClientResponse
from ..storage import StoredPayload
from .contracts import (
    ModelClientBodyRef,
    ModelClientRequestRef,
    ModelClientResponseRef,
    RuntimePayloadRef,
)
from ._archive import ArchiveRoot, PreparedArchive
from ._payload import decode_payload_bytes
from ._plan import PersistenceDomain

if TYPE_CHECKING:
    from ._history import TranscriptRepository


async def prepare_client_requests(
    requests: tuple[ModelClientRequest, ...] | None,
    runtime_domain: PersistenceDomain,
    history: TranscriptRepository | None = None,
    *, prepared: PreparedArchive | None = None,
) -> tuple[ModelClientRequestRef, ...] | None:
    if requests is None:
        return None

    async def prepare_body(raw: bytes | None) -> ModelClientBodyRef | None:
        if raw is None or runtime_domain is not PersistenceDomain.TRACE:
            return None
        if history is None:
            content = RuntimePayloadRef(StoredPayload.inline_bytes(raw), runtime_domain)
        else:
            if prepared is None:
                raise ValueError("durable client bodies require the owner's prepared archive")
            archive = await history.content_archive(create=True)
            content = await archive.prepare_append(archive.empty_root(), (("model-client-body", raw),), prepared)
        return ModelClientBodyRef("raw", hashlib.sha256(raw).hexdigest(), len(raw), content)

    result: list[ModelClientRequestRef] = []
    for request in requests:
        response = request.response
        result.append(ModelClientRequestRef(
            api=request.api,
            record_http_bodies=request.record_http_bodies,
            diagnostics=request.diagnostics,
            body=await prepare_body(request.body),
            capture_error=request.capture_error,
            response=None if response is None else ModelClientResponseRef(
                body=await prepare_body(response.body),
                status_code=response.status_code,
                format=response.format,
                content_encoding=response.content_encoding,
                complete=response.complete,
                capture_error=response.capture_error,
            ),
        ))
    return tuple(result)


async def capture_client_requests(
    requests: tuple[ModelClientRequestRef, ...] | None,
    runtime_domain: PersistenceDomain,
    history: TranscriptRepository | None = None,
    *, include_bodies: bool = True,
) -> Callable[[], Awaitable[tuple[ModelClientRequest, ...] | None]]:
    """Capture database bodies with their owner; decode after its read scope closes."""
    roots = tuple(dict.fromkeys(
        body.content for request in requests or ()
        for body in (request.body, None if request.response is None else request.response.body)
        if include_bodies and body is not None and isinstance(body.content, ArchiveRoot)
    ))
    load_ranges = None
    if roots:
        if history is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if runtime_domain is not PersistenceDomain.TRACE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = await history.content_archive()
        load_ranges = await archive.capture_ranges(roots, codec="model-client-body")

    async def load() -> tuple[ModelClientRequest, ...] | None:
        if requests is None:
            return None
        values = await load_ranges() if load_ranges is not None else ()
        archived = {root: group[0] for root, group in zip(roots, values, strict=True)}

        async def read_body(body: ModelClientBodyRef | None) -> bytes | None:
            if body is None or not include_bodies:
                return None
            if isinstance(body.content, ArchiveRoot):
                encoded = archived[body.content]
            else:
                if body.content.source_domain not in {None, runtime_domain}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = body.content.payload
                if history is None:
                    if payload.kind != "inline":
                        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                    try:
                        encoded = payload.decode()
                    except (TypeError, ValueError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    if not isinstance(encoded, bytes):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                else:
                    encoded = await history.read_payload(payload)
            return decode_payload_bytes(
                encoded, codec=body.codec, raw_digest=body.raw_digest, raw_size=body.raw_size,
            )

        result: list[ModelClientRequest] = []
        for request in requests:
            response = request.response
            result.append(ModelClientRequest(
                api=request.api,
                record_http_bodies=request.record_http_bodies,
                diagnostics=request.diagnostics,
                body=await read_body(request.body),
                capture_error=request.capture_error,
                response=None if response is None else ModelClientResponse(
                    body=await read_body(response.body),
                    status_code=response.status_code,
                    format=cast(Literal["json", "sse", "unknown"], response.format),
                    content_encoding=response.content_encoding,
                    complete=response.complete,
                    capture_error=response.capture_error,
                ),
            ))
        return tuple(result)

    return load


async def resolve_client_requests(
    requests: tuple[ModelClientRequestRef, ...] | None,
    runtime_domain: PersistenceDomain,
    history: TranscriptRepository | None = None,
    *, include_bodies: bool = True,
) -> tuple[ModelClientRequest, ...] | None:
    load = await capture_client_requests(requests, runtime_domain, history, include_bodies=include_bodies)
    return await load()


__all__ = ["prepare_client_requests", "capture_client_requests", "resolve_client_requests"]
