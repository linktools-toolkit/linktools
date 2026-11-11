#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compression and decoded integrity shared by Runtime history payloads."""

import hashlib
import zlib

from ..errors import AIError, ErrorCode

PAYLOAD_COMPRESS_MINIMUM = 16 * 1024


def encode_payload_bytes(raw: bytes) -> tuple[str, bytes]:
    content = raw
    codec = "raw"
    if len(raw) >= PAYLOAD_COMPRESS_MINIMUM:
        compressed = zlib.compress(raw)
        if len(compressed) <= len(raw) * 0.9:
            content = compressed
            codec = "zlib"
    return codec, content


def decode_payload_bytes(
    content: bytes, *, codec: str, raw_digest: str, raw_size: int,
) -> bytes:
    if type(raw_size) is not int or raw_size < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if codec == "zlib":
        try:
            decoder = zlib.decompressobj()
            # The extra byte distinguishes a false declared size without ever
            # materializing the rest of an oversized compressed payload.
            content = decoder.decompress(content, raw_size + 1)
            if (len(content) > raw_size or not decoder.eof
                    or decoder.unused_data or decoder.unconsumed_tail):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except (zlib.error, OverflowError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    elif codec != "raw":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if len(content) != raw_size or hashlib.sha256(content).hexdigest() != raw_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content


__all__ = ["PAYLOAD_COMPRESS_MINIMUM", "encode_payload_bytes", "decode_payload_bytes"]
