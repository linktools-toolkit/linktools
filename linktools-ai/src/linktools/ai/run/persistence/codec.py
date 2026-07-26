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
from typing import TYPE_CHECKING

from .wire import (
    SCHEMA_VERSION,
    RunCommitCodecError,
    RunCommitIntegrityError,
    RunCommitOperation,
    decode_envelope,
    encode_envelope,
)
from ..commit import (
    AcknowledgeCancelRunCommand, CompleteRunCommand, FailRunCommand,
    PauseRunCommand, RequestCancelRunCommand, ResumeRunCommand, StartRunCommand,
    CancellingRunCommit, CancelledRunCommit, CompletedRunCommit, FailedRunCommit,
    PausedRunCommit, ResumedRunCommit, StartedRunCommit,
)

RunCommitCommand = (StartRunCommand | PauseRunCommand | ResumeRunCommand |
                    CompleteRunCommand | FailRunCommand | RequestCancelRunCommand |
                    AcknowledgeCancelRunCommand)
RunCommitResult = (StartedRunCommit | PausedRunCommit | ResumedRunCommit |
                   CompletedRunCommit | FailedRunCommit | CancellingRunCommit |
                   CancelledRunCommit)

__all__ = [
    "RunCommitCodec",
    "RunCommitCodecError",
    "RunCommitIntegrityError",
    "RunCommitOperation",
    "RunCommitCommand",
    "RunCommitResult",
]


class RunCommitCodec:
    """Encode/decode the seven Run commit operations' request and result
    payloads. Stateless and safe to share across coordinators."""

    schema_version = SCHEMA_VERSION

    def encode_request(
        self,
        operation: "RunCommitOperation | str",
        command: "RunCommitCommand",
    ) -> bytes:
        return encode_envelope(operation, command, kind="request")

    def decode_request(
        self,
        operation: "RunCommitOperation | str",
        payload: bytes,
    ) -> "RunCommitCommand":
        return decode_envelope(payload, expected_operation=operation, expected_kind="request")

    def encode_result(
        self,
        operation: "RunCommitOperation | str",
        result: "RunCommitResult",
    ) -> bytes:
        return encode_envelope(operation, result, kind="result")

    def decode_result(
        self,
        operation: "RunCommitOperation | str",
        payload: bytes,
    ) -> "RunCommitResult":
        return decode_envelope(payload, expected_operation=operation, expected_kind="result")

    def request_hash(
        self,
        operation: "RunCommitOperation | str",
        command: "RunCommitCommand",
    ) -> bytes:
        return sha256(self.encode_request(operation, command)).digest()

    def result_hash(
        self,
        operation: "RunCommitOperation | str",
        result: "RunCommitResult",
    ) -> bytes:
        return sha256(self.encode_result(operation, result)).digest()
