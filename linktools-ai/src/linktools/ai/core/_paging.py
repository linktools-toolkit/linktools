#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded pages and authenticated cursors."""

import base64
import binascii
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ..errors import AIError, ErrorCode
from ._json import JsonValue, canonical_json_bytes

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class Page(Generic[ItemT]):
    items: "tuple[ItemT, ...]"
    next_cursor: "str | None" = None


@dataclass(frozen=True, slots=True)
class CursorPayload:
    cursor_version: int
    tenant_id: str
    resource_kind: str
    filter_digest: str
    sort_key: str
    snapshot_or_store_revision: int
    expires_at: int
    include_deleted: bool = False

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "cursor_version": self.cursor_version,
            "tenant_id": self.tenant_id,
            "resource_kind": self.resource_kind,
            "filter_digest": self.filter_digest,
            "sort_key": self.sort_key,
            "snapshot_or_store_revision": self.snapshot_or_store_revision,
            "expires_at": self.expires_at,
            "include_deleted": self.include_deleted,
        }


class CursorSigner(Protocol):
    def encode(self, payload: CursorPayload) -> str: ...
    def decode(self, token: str) -> CursorPayload: ...


class HmacCursorSigner:
    """Sign canonical cursors with an injected current and previous key."""

    def __init__(self, current_key_id: str, current_key: bytes, previous: "tuple[str, bytes] | None" = None) -> None:
        if not current_key_id or "." in current_key_id or not current_key:
            raise ValueError("cursor signing key is required")
        self._current_key_id = current_key_id
        self._keys = {current_key_id: current_key}
        if previous is not None:
            if not previous[0] or "." in previous[0] or not previous[1]:
                raise ValueError("previous cursor signing key is invalid")
            self._keys[previous[0]] = previous[1]

    def encode(self, payload: CursorPayload) -> str:
        raw = canonical_json_bytes(payload.as_json())
        key_id = self._current_key_id.encode("utf-8")
        signature = hmac.new(self._keys[self._current_key_id], raw + b"." + key_id, hashlib.sha256).digest()
        return ".".join((_b64(raw), self._current_key_id, _b64(signature)))

    def decode(self, token: str) -> CursorPayload:
        try:
            raw_token, key_id, encoded_signature = token.split(".", 2)
            raw = _unb64(raw_token)
            signature = _unb64(encoded_signature)
            key = self._keys[key_id]
            expected = hmac.new(key, raw + b"." + key_id.encode("utf-8"), hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("include_deleted"), bool):
                raise ValueError("cursor fields")
            payload = CursorPayload(
                int(value["cursor_version"]),
                str(value["tenant_id"]),
                str(value["resource_kind"]),
                str(value["filter_digest"]),
                str(value["sort_key"]),
                int(value["snapshot_or_store_revision"]),
                int(value["expires_at"]),
                value["include_deleted"],
            )
            if payload.expires_at < int(time.time()):
                raise ValueError("expired")
            return payload
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError, binascii.Error, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.ASSET_CURSOR_INVALID) from error


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


__all__ = ["CursorPayload", "CursorSigner", "HmacCursorSigner", "Page"]
