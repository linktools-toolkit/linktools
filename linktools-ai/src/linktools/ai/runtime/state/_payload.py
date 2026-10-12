#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compression and decoded integrity shared by Runtime history payloads."""

import hashlib
import zlib

from ...errors import AIError, ErrorCode

PAYLOAD_COMPRESS_MINIMUM = 16 * 1024


def encode_payload_bytes(raw: bytes) -> tuple[str, str, int, bytes]:
    content = raw
    codec = "raw"
    if len(raw) >= PAYLOAD_COMPRESS_MINIMUM:
        compressed = zlib.compress(raw)
        if len(compressed) <= len(raw) * 0.9:
            content = compressed
            codec = "zlib"
    return codec, hashlib.sha256(raw).hexdigest(), len(raw), content


def decode_payload_bytes(
    content: bytes, *, codec: str, raw_digest: str, raw_size: int,
) -> bytes:
    if codec == "zlib":
        try:
            content = zlib.decompress(content)
        except zlib.error as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    elif codec != "raw":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if len(content) != raw_size or hashlib.sha256(content).hexdigest() != raw_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content


__all__ = ["PAYLOAD_COMPRESS_MINIMUM", "encode_payload_bytes", "decode_payload_bytes"]
