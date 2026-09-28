#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime cursor encoding and common identity checks."""

import time

from ..core import CursorPayload, CursorSigner
from ..errors import AIError, ErrorCode

_CURSOR_TTL_SECONDS = 3600


def encode_cursor(
    signer: CursorSigner,
    *,
    tenant_id: str,
    resource_kind: str,
    filter_digest: str,
    position: str,
    revision: int = 0,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            resource_kind,
            filter_digest,
            position,
            revision,
            int(time.time()) + _CURSOR_TTL_SECONDS,
        )
    )


def decode_cursor(
    token: str,
    signer: CursorSigner,
    *,
    tenant_id: str,
    resource_kind: str,
    filter_digest: str,
) -> CursorPayload:
    try:
        payload = signer.decode(token)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.tenant_id != tenant_id
        or payload.resource_kind != resource_kind
        or payload.filter_digest != filter_digest
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return payload


__all__: list[str] = []
