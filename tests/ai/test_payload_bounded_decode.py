#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Payload decoding obeys its declared output bound and exact stream boundary."""

import hashlib
import zlib

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._payload import decode_payload_bytes, encode_payload_bytes


@pytest.mark.parametrize("raw", (b"", b"payload", b"repeat" * 10000))
def test_payload_roundtrip_preserves_empty_raw_and_compressed_bytes(raw: bytes) -> None:
    codec, content = encode_payload_bytes(raw)
    digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
    assert decode_payload_bytes(content, codec=codec, raw_digest=digest, raw_size=size) == raw
    assert decode_payload_bytes(zlib.compress(raw), codec="zlib", raw_digest=digest, raw_size=size) == raw


@pytest.mark.parametrize("damage", ("oversize", "truncated", "trailing", "concatenated"))
def test_compressed_payload_rejects_excess_output_and_nonexact_stream(damage: str) -> None:
    raw = b"a" * (1024 * 1024)
    content = zlib.compress(raw)
    size = len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if damage == "oversize":
        size = 8
        digest = hashlib.sha256(raw[:size]).hexdigest()
    elif damage == "truncated":
        content = content[:-1]
    elif damage == "trailing":
        content += b"unowned trailing data"
    elif damage == "concatenated":
        content += zlib.compress(b"another stream")
    with pytest.raises(AIError) as failure:
        decode_payload_bytes(content, codec="zlib", raw_digest=digest, raw_size=size)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("size", (-1, True, 1.0))
def test_payload_size_requires_a_nonnegative_integer(size: object) -> None:
    with pytest.raises(AIError) as failure:
        decode_payload_bytes(zlib.compress(b""), codec="zlib",
                             raw_digest=hashlib.sha256(b"").hexdigest(), raw_size=size)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
