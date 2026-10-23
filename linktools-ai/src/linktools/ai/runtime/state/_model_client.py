#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original model client bytes in the existing interaction payload lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from ...errors import AIError, ErrorCode
from ...model import ModelClientRequest, ModelClientResponse
from ...storage import StoredPayload
from ._contracts import (
    ModelClientBodyRef,
    ModelClientRequestRef,
    ModelClientResponseRef,
    RuntimePayloadRef,
)
from ._payload import decode_payload_bytes, encode_payload_bytes
from ._plan import RuntimeDomain

if TYPE_CHECKING:
    from ._history import TranscriptRepository


async def prepare_client_requests(
    requests: tuple[ModelClientRequest, ...] | None,
    runtime_domain: RuntimeDomain,
    history: TranscriptRepository | None = None,
) -> tuple[ModelClientRequestRef, ...] | None:
    if requests is None:
        return None

    async def prepare_body(raw: bytes | None) -> ModelClientBodyRef | None:
        if raw is None or runtime_domain is not RuntimeDomain.TRACE:
            return None
        codec, raw_digest, raw_size, encoded = encode_payload_bytes(raw)
        payload = (
            StoredPayload.inline_bytes(encoded)
            if history is None
            else await history.store_payload(encoded, raw_size=raw_size)
        )
        return ModelClientBodyRef(
            codec, raw_digest, raw_size, RuntimePayloadRef(payload, runtime_domain),
        )

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


async def resolve_client_requests(
    requests: tuple[ModelClientRequestRef, ...] | None,
    runtime_domain: RuntimeDomain,
    history: TranscriptRepository | None = None,
    *, include_bodies: bool = True,
) -> tuple[ModelClientRequest, ...] | None:
    if requests is None:
        return None

    async def read_body(body: ModelClientBodyRef | None) -> bytes | None:
        if body is None or not include_bodies:
            return None
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


__all__ = ["prepare_client_requests", "resolve_client_requests"]
