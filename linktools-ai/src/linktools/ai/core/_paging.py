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
_CURSOR_FIELDS = frozenset(
    {
        "version",
        "tenant_id",
        "resource_kind",
        "filter_digest",
        "position",
        "revision",
        "expires_at",
    }
)


@dataclass(frozen=True, slots=True)
class Page(Generic[ItemT]):
    items: "tuple[ItemT, ...]"
    next_cursor: "str | None" = None


@dataclass(frozen=True, slots=True)
class CursorPayload:
    version: int
    tenant_id: str
    resource_kind: str
    filter_digest: str
    position: str
    revision: int
    expires_at: int

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("unsupported cursor version")
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise ValueError("cursor tenant is invalid")
        if not isinstance(self.resource_kind, str) or not self.resource_kind:
            raise ValueError("cursor resource kind is invalid")
        if (
            not isinstance(self.filter_digest, str)
            or len(self.filter_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.filter_digest
            )
        ):
            raise ValueError("cursor filter digest is invalid")
        if not isinstance(self.position, str) or not self.position:
            raise ValueError("cursor position is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("cursor revision is invalid")
        if (
            not isinstance(self.expires_at, int)
            or isinstance(self.expires_at, bool)
            or self.expires_at <= 0
        ):
            raise ValueError("cursor expiry is invalid")

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "tenant_id": self.tenant_id,
            "resource_kind": self.resource_kind,
            "filter_digest": self.filter_digest,
            "position": self.position,
            "revision": self.revision,
            "expires_at": self.expires_at,
        }


class CursorSigner(Protocol):
    def encode(self, payload: CursorPayload) -> str: ...
    def decode(self, token: str) -> CursorPayload: ...


class HmacCursorSigner:
    """Sign canonical cursors with an injected current and previous key."""

    def __init__(
        self,
        current_key_id: str,
        current_key: bytes,
        previous: "tuple[str, bytes] | None" = None,
    ) -> None:
        if not current_key_id or "." in current_key_id or not current_key:
            raise ValueError("cursor signing key is required")
        self._current_key_id = current_key_id
        self._keys = {current_key_id: current_key}
        if previous is not None:
            if (
                not previous[0]
                or "." in previous[0]
                or not previous[1]
                or previous[0] == current_key_id
            ):
                raise ValueError("previous cursor signing key is invalid")
            self._keys[previous[0]] = previous[1]

    def encode(self, payload: CursorPayload) -> str:
        raw = canonical_json_bytes(payload.as_json())
        key_id = self._current_key_id.encode("utf-8")
        signature = hmac.new(
            self._keys[self._current_key_id],
            raw + b"." + key_id,
            hashlib.sha256,
        ).digest()
        return ".".join((_b64(raw), self._current_key_id, _b64(signature)))

    def decode(self, token: str) -> CursorPayload:
        try:
            if not isinstance(token, str):
                raise TypeError("cursor token must be a string")
            raw_token, key_id, encoded_signature = token.split(".", 2)
            raw = _unb64(raw_token)
            signature = _unb64(encoded_signature)
            key = self._keys[key_id]
            expected = hmac.new(
                key,
                raw + b"." + key_id.encode("utf-8"),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("cursor signature is invalid")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or set(value) != _CURSOR_FIELDS:
                raise ValueError("cursor fields are invalid")
            payload = CursorPayload(
                value["version"],
                value["tenant_id"],
                value["resource_kind"],
                value["filter_digest"],
                value["position"],
                value["revision"],
                value["expires_at"],
            )
            if payload.expires_at < int(time.time()):
                raise ValueError("cursor is expired")
            return payload
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
            binascii.Error,
            json.JSONDecodeError,
        ) as error:
            raise AIError(ErrorCode.CURSOR_INVALID) from error


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


__all__ = ["CursorPayload", "CursorSigner", "HmacCursorSigner", "Page"]
