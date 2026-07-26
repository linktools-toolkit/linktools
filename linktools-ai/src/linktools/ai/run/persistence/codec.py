#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single canonical wire codec for Run durable commit requests/results.

The codec is a thin, operation-validated dispatcher over the typed wire
engine in :mod:`linktools.ai.run.persistence.wire`. Every request and result
is encoded as a self-describing, type-tagged envelope that decodes back into
the EXACT domain dataclass (StartedRunCommit / CompletedRunCommit / ...), so
a replay returns the first persisted typed result rather than a bare dict or
a freshly-rebuilt value from the current command.

Bytes stability (and therefore ``request_hash``) is identical across the
Filesystem and SQL backends and across ``PYTHONHASHSEED`` values: the wire
engine canonicalises dict key order and set iteration order before hashing."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from .wire import (
    SCHEMA_VERSION,
    RunCommitCodecError,
    RunCommitIntegrityError,
    RunCommitOperation,
    decode_envelope,
    encode_envelope,
)

__all__ = [
    "RunCommitCodec",
    "RunCommitCodecError",
    "RunCommitIntegrityError",
    "RunCommitOperation",
]


class RunCommitCodec:
    """Encode/decode the seven Run commit operations' request and result
    payloads. Stateless and safe to share across coordinators."""

    schema_version = SCHEMA_VERSION

    def encode_request(
        self,
        operation: "RunCommitOperation | str",
        command: Any,
    ) -> bytes:
        return encode_envelope(operation, command)

    def decode_request(
        self,
        operation: "RunCommitOperation | str",
        payload: bytes,
    ) -> Any:
        return decode_envelope(payload, expected_operation=operation)

    def encode_result(
        self,
        operation: "RunCommitOperation | str",
        result: Any,
    ) -> bytes:
        return encode_envelope(operation, result)

    def decode_result(
        self,
        operation: "RunCommitOperation | str",
        payload: bytes,
    ) -> Any:
        return decode_envelope(payload, expected_operation=operation)

    def request_hash(
        self,
        operation: "RunCommitOperation | str",
        command: Any,
    ) -> bytes:
        return sha256(self.encode_request(operation, command)).digest()

    def result_hash(
        self,
        operation: "RunCommitOperation | str",
        result: Any,
    ) -> bytes:
        return sha256(self.encode_result(operation, result)).digest()
