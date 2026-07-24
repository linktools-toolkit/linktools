#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-backend Asset list page token: a versioned, opaque, tamper-proof
cursor that carries EACH backend's own pagination position (never a single
combined "furthest scanned path" string). A single shared cursor position
would force a backend that has only scanned up to path M to jump straight to
another backend's furthest-scanned path Z on the next call, silently skipping
every one of its own real items between M and Z -- the exact defect this
cursor shape exists to prevent.

Encoding: canonical JSON (sorted keys, fixed separators, so two encodings of
the same state always produce the same bytes) -> UTF-8 -> URL-safe base64 ->
HMAC-SHA256 tag appended, so a caller cannot forge or replay a modified cursor
without the server-held secret. ``AssetStore`` cross-checks the DECODED
cursor's backend ids/count/revision against the LIVE backend set on every
call (see :meth:`AssetStore.list`) -- the codec itself only proves the token
was not tampered with; it has no way to know what backends currently exist."""

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Literal

from ..errors import InvalidAssetCursorError

# Caps (fail closed, no silent truncation): a cursor referencing more backends
# than a sane multi-backend AssetStore composition, or exceeding the decoded
# byte budget, is rejected outright rather than partially honored.
MAX_CURSOR_BACKENDS = 32
MAX_DECODED_CURSOR_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class BufferedAssetHead:
    """One buffered candidate from a single backend's pagination stream --
    NOT yet output, just held so the k-way merge can compare heads across
    backends. ``whiteout`` marks a tombstone (the primary deleted this path);
    a tombstone still occupies a path slot in the merge so an overlay's live
    entry at the same path is correctly shadowed rather than resurrected."""

    path: str
    kind: str
    version: "int | None"
    etag: "str | None"
    whiteout: bool


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
    buffered: "tuple[BufferedAssetHead, ...]"
    exhausted: bool
    revision: str


@dataclass(frozen=True, slots=True)
class AssetListCursor:
    version: "Literal[1]"
    backend_states: "tuple[BackendCursorState, ...]"


def _encode_head(head: BufferedAssetHead) -> "dict[str, object]":
    return {
        "path": head.path,
        "kind": head.kind,
        "version": head.version,
        "etag": head.etag,
        "whiteout": head.whiteout,
    }


def _decode_head(raw: object) -> BufferedAssetHead:
    if not isinstance(raw, dict):
        raise InvalidAssetCursorError("malformed buffered head in cursor")
    try:
        return BufferedAssetHead(
            path=raw["path"],
            kind=raw["kind"],
            version=raw["version"],
            etag=raw["etag"],
            whiteout=bool(raw["whiteout"]),
        )
    except KeyError as exc:
        raise InvalidAssetCursorError(f"buffered head missing field: {exc}") from None


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
        raise InvalidAssetCursorError("malformed backend cursor state")
    try:
        buffered_raw = raw["buffered"]
        if not isinstance(buffered_raw, list):
            raise InvalidAssetCursorError("backend cursor state 'buffered' must be a list")
        return BackendCursorState(
            backend_id=raw["backend_id"],
            cursor=raw["cursor"],
            buffered=tuple(_decode_head(h) for h in buffered_raw),
            exhausted=bool(raw["exhausted"]),
            revision=raw["revision"],
        )
    except KeyError as exc:
        raise InvalidAssetCursorError(f"backend cursor state missing field: {exc}") from None


class AssetCursorCodec:
    """Encodes/decodes the opaque page token. ``secret`` must be the SAME
    across every process that needs to decode a token another process
    minted (a multi-process downstream shares one secret); a single-process
    deployment may use a fresh random secret per process since it never
    needs to decode a cursor minted by a different process."""

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def encode(self, cursor: AssetListCursor) -> str:
        if cursor.version != 1:
            raise InvalidAssetCursorError(f"unsupported cursor version: {cursor.version!r}")
        if len(cursor.backend_states) > MAX_CURSOR_BACKENDS:
            raise InvalidAssetCursorError(
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

    def decode(self, token: str) -> AssetListCursor:
        parts = token.split(".")
        if len(parts) != 2:
            raise InvalidAssetCursorError("malformed cursor token")
        body_str, tag_str = parts
        body = body_str.encode("ascii")
        expected_tag = hmac.new(self._secret, body, hashlib.sha256).digest()
        try:
            actual_tag = base64.urlsafe_b64decode(_pad(tag_str))
        except Exception:
            raise InvalidAssetCursorError("malformed cursor tag") from None
        # Constant-time compare -- a cursor tag is a MAC verification, not a
        # value lookup; a timing side channel would let an attacker forge a
        # valid tag byte-by-byte.
        if not hmac.compare_digest(expected_tag, actual_tag):
            raise InvalidAssetCursorError("cursor tag mismatch (tampered or wrong secret)")
        try:
            raw = base64.urlsafe_b64decode(_pad(body_str))
        except Exception:
            raise InvalidAssetCursorError("malformed cursor body") from None
        if len(raw) > MAX_DECODED_CURSOR_BYTES:
            raise InvalidAssetCursorError(
                f"decoded cursor exceeds {MAX_DECODED_CURSOR_BYTES} bytes"
            )
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise InvalidAssetCursorError("malformed cursor JSON") from None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise InvalidAssetCursorError(
                f"unsupported cursor version: {payload.get('version') if isinstance(payload, dict) else None!r}"
            )
        states_raw = payload.get("backend_states")
        if not isinstance(states_raw, list) or len(states_raw) > MAX_CURSOR_BACKENDS:
            raise InvalidAssetCursorError("invalid backend_states in cursor")
        return AssetListCursor(
            version=1, backend_states=tuple(_decode_state(s) for s in states_raw)
        )


def _pad(b64_str: str) -> str:
    return b64_str + "=" * (-len(b64_str) % 4)


__all__: "list[str]" = [
    "BufferedAssetHead",
    "BackendCursorState",
    "AssetListCursor",
    "AssetCursorCodec",
    "MAX_CURSOR_BACKENDS",
    "MAX_DECODED_CURSOR_BYTES",
]
