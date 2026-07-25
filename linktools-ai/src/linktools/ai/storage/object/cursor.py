#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-backend Object list page token: a versioned, opaque, tamper-proof
cursor that carries EACH backend's own pagination position (never a single
combined "furthest scanned key" string). A single shared cursor position
would force a backend that has only scanned up to key M to jump straight to
another backend's furthest-scanned key Z on the next call, silently skipping
every one of its own real items between M and Z -- the exact defect this
cursor shape exists to prevent.

Encoding: canonical JSON (sorted keys, fixed separators, so two encodings of
the same state always produce the same bytes) -> UTF-8 -> URL-safe base64 ->
HMAC-SHA256 tag appended, so a caller cannot forge or replay a modified cursor
without the server-held secret. ``OverlayObjectStore`` cross-checks the
DECODED cursor's backend ids/count/revision against the LIVE backend set on
every call -- the codec itself only proves the token was not tampered with; it
has no way to know what backends currently exist."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..features import CoordinationScope
from .errors import StorageObjectError

# Caps (fail closed, no silent truncation): a cursor referencing more backends
# than a sane multi-backend composition, or exceeding the decoded byte
# budget, is rejected outright rather than partially honored.
MAX_CURSOR_BACKENDS = 32
MAX_DECODED_CURSOR_BYTES = 256 * 1024


class InvalidObjectCursorError(StorageObjectError):
    """The cursor token failed to decode (tampered, malformed, wrong secret,
    or an unsupported version)."""


class StaleObjectCursorError(StorageObjectError):
    """The cursor's backend set or a per-backend revision no longer matches
    the live composition -- a resumed listing is never silently continued
    against inconsistent backend state."""


@dataclass(frozen=True, slots=True)
class BufferedObjectHead:
    """One buffered candidate from a single backend's pagination stream --
    NOT yet output, just held so the k-way merge can compare heads across
    backends. ``tombstone`` marks a deleted key; a tombstone still occupies a
    key slot in the merge so an overlay's live entry at the same key is
    correctly shadowed rather than resurrected."""

    key: str
    version: "int | None"
    etag: "str | None"
    tombstone: bool


@dataclass(frozen=True, slots=True)
class BackendCursorState:
    """One backend's complete pagination position: its own opaque page
    cursor (or None if it has never been queried / has no more pages), any
    heads already fetched but not yet consumed by the merge, whether it is
    exhausted, and the backend's revision AT THE TIME this state was minted
    (so a later revision change is detected as staleness, not silently
    ignored)."""

    backend_id: str
    cursor: "str | None"
    buffered: "tuple[BufferedObjectHead, ...]"
    exhausted: bool
    revision: str


@dataclass(frozen=True, slots=True)
class ObjectListCursor:
    version: "Literal[1]"
    backend_states: "tuple[BackendCursorState, ...]"


def _encode_head(head: BufferedObjectHead) -> "dict[str, object]":
    return {
        "key": head.key,
        "version": head.version,
        "etag": head.etag,
        "tombstone": head.tombstone,
    }


def _decode_head(raw: object) -> BufferedObjectHead:
    if not isinstance(raw, dict):
        raise InvalidObjectCursorError("malformed buffered head in cursor")
    try:
        return BufferedObjectHead(
            key=raw["key"],
            version=raw["version"],
            etag=raw["etag"],
            tombstone=bool(raw["tombstone"]),
        )
    except KeyError as exc:
        raise InvalidObjectCursorError(f"buffered head missing field: {exc}") from None


def _encode_state(state: BackendCursorState) -> "dict[str, object]":
    return {
        "backend_id": state.backend_id,
        "cursor": state.cursor,
        "buffered": [_encode_head(h) for h in state.buffered],
        "exhausted": state.exhausted,
        "revision": state.revision,
    }


def _decode_state(raw: object) -> BackendCursorState:
    if not isinstance(raw, dict):
        raise InvalidObjectCursorError("malformed backend cursor state")
    try:
        buffered_raw = raw["buffered"]
        if not isinstance(buffered_raw, list):
            raise InvalidObjectCursorError("backend cursor state 'buffered' must be a list")
        return BackendCursorState(
            backend_id=raw["backend_id"],
            cursor=raw["cursor"],
            buffered=tuple(_decode_head(h) for h in buffered_raw),
            exhausted=bool(raw["exhausted"]),
            revision=raw["revision"],
        )
    except KeyError as exc:
        raise InvalidObjectCursorError(f"backend cursor state missing field: {exc}") from None


@runtime_checkable
class ObjectCursorCodecProtocol(Protocol):
    """Encodes/decodes the opaque Object list-page token and DECLARES the
    coordination scope the secret is valid for.

    ``scope`` is what makes multi-worker safety a CONSTRUCTION gate rather
    than a silent default: a process-local secret (random per process) is
    only decodable inside that one process, so a codec built for a single
    process declares ``PROCESS_LOCAL`` and the Runtime build gate refuses it
    under a MULTI_WORKER topology that shares Object listings across workers;
    a multi-process deployment injects a codec built from a shared secret
    and declares ``DISTRIBUTED``."""

    @property
    def scope(self) -> "CoordinationScope":
        ...

    def encode(self, cursor: "ObjectListCursor") -> str:
        ...

    def decode(self, token: str) -> "ObjectListCursor":
        ...


class HmacObjectCursorCodec:
    """Encodes/decodes the opaque page token. ``secret`` must be the SAME
    across every process that needs to decode a token another process
    minted (a multi-process downstream shares one secret); a single-process
    deployment may use a fresh random secret per process since it never
    needs to decode a cursor minted by a different process.

    ``scope`` declares the range that shared secret actually covers, so the
    Runtime build gate can refuse a process-local codec under a topology
    that shares Object listings across workers."""

    def __init__(self, secret: bytes, *, scope: "CoordinationScope") -> None:
        # 32 bytes = 256-bit: anything shorter weakens the HMAC against
        # brute-force forgery. Refused outright rather than silently padded.
        if len(secret) < 32:
            raise ValueError("object cursor HMAC secret must be at least 32 bytes")
        self._secret = secret
        self._scope = scope

    @property
    def scope(self) -> "CoordinationScope":
        return self._scope

    def encode(self, cursor: ObjectListCursor) -> str:
        if cursor.version != 1:
            raise InvalidObjectCursorError(f"unsupported cursor version: {cursor.version!r}")
        if len(cursor.backend_states) > MAX_CURSOR_BACKENDS:
            raise InvalidObjectCursorError(
                f"cursor references {len(cursor.backend_states)} backends, "
                f"exceeding the cap of {MAX_CURSOR_BACKENDS}"
            )
        payload = {
            "version": cursor.version,
            "backend_states": [_encode_state(s) for s in cursor.backend_states],
        }
        # Fixed key order + separators: two encodings of equal state always
        # produce identical bytes, so the HMAC tag is deterministic.
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        body = base64.urlsafe_b64encode(raw).rstrip(b"=")
        tag = hmac.new(self._secret, body, hashlib.sha256).digest()
        tag_b64 = base64.urlsafe_b64encode(tag).rstrip(b"=")
        return f"{body.decode('ascii')}.{tag_b64.decode('ascii')}"

    def decode(self, token: str) -> ObjectListCursor:
        parts = token.split(".")
        if len(parts) != 2:
            raise InvalidObjectCursorError("malformed cursor token")
        body_str, tag_str = parts
        body = body_str.encode("ascii")
        expected_tag = hmac.new(self._secret, body, hashlib.sha256).digest()
        try:
            actual_tag = base64.urlsafe_b64decode(_pad(tag_str))
        except Exception:
            raise InvalidObjectCursorError("malformed cursor tag") from None
        # Constant-time compare -- a cursor tag is a MAC verification, not a
        # value lookup; a timing side channel would let an attacker forge a
        # valid tag byte-by-byte.
        if not hmac.compare_digest(expected_tag, actual_tag):
            raise InvalidObjectCursorError("cursor tag mismatch (tampered or wrong secret)")
        try:
            raw = base64.urlsafe_b64decode(_pad(body_str))
        except Exception:
            raise InvalidObjectCursorError("malformed cursor body") from None
        if len(raw) > MAX_DECODED_CURSOR_BYTES:
            raise InvalidObjectCursorError(
                f"decoded cursor exceeds {MAX_DECODED_CURSOR_BYTES} bytes"
            )
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise InvalidObjectCursorError("malformed cursor JSON") from None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise InvalidObjectCursorError(
                f"unsupported cursor version: "
                f"{payload.get('version') if isinstance(payload, dict) else None!r}"
            )
        states_raw = payload.get("backend_states")
        if not isinstance(states_raw, list) or len(states_raw) > MAX_CURSOR_BACKENDS:
            raise InvalidObjectCursorError("invalid backend_states in cursor")
        return ObjectListCursor(
            version=1, backend_states=tuple(_decode_state(s) for s in states_raw)
        )


def _pad(b64_str: str) -> str:
    return b64_str + "=" * (-len(b64_str) % 4)


__all__: "list[str]" = [
    "BufferedObjectHead",
    "BackendCursorState",
    "ObjectListCursor",
    "ObjectCursorCodecProtocol",
    "HmacObjectCursorCodec",
    "InvalidObjectCursorError",
    "StaleObjectCursorError",
    "MAX_CURSOR_BACKENDS",
    "MAX_DECODED_CURSOR_BYTES",
]
