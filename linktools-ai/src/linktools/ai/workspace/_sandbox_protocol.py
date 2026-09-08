#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded framing shared by the Bubblewrap host and worker."""

from __future__ import annotations

import asyncio
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode

PROTOCOL_VERSION = 1
WORKER_BUILD = "linktools-ai-sandbox-worker-v1"
GUARDIAN_EXIT_OK = 0
GUARDIAN_EXIT_SESSION_FAILED = 1
GUARDIAN_EXIT_CLEANUP_FAILED = 2
WORKER_EXIT_OK = 0
WORKER_EXIT_SESSION_FAILED = 1
WORKER_EXIT_CLEANUP_FAILED = 2
MAX_FRAME_BYTES = 8 * 1024 * 1024
_SIZE_CHECK_REQUEST_ID = "0" * 32
_REQUEST_FIELDS = {
    "read_file": (
        frozenset({"path"}),
        frozenset({"offset", "limit"}),
    ),
    "write_file": (
        frozenset({"path", "content"}),
        frozenset({"expected_hash"}),
    ),
    "edit_file": (
        frozenset({"path", "old_text", "new_text"}),
        frozenset({"expected_hash"}),
    ),
    "list_directory": (frozenset(), frozenset({"path"})),
    "search_files": (
        frozenset({"pattern"}),
        frozenset({"path", "include_glob"}),
    ),
    "find_files": (
        frozenset({"pattern"}),
        frozenset({"path"}),
    ),
    "create_directory": (frozenset({"path"}), frozenset()),
    "file_info": (frozenset({"path"}), frozenset()),
    "run_command": (
        frozenset({"command"}),
        frozenset({"timeout_seconds"}),
    ),
    "start_command": (frozenset({"command"}), frozenset()),
    "check_command": (frozenset({"command_id"}), frozenset()),
    "stop_command": (frozenset({"command_id"}), frozenset()),
}
_REQUEST_VALUE_TYPES = {
    "path": (str,),
    "pattern": (str,),
    "include_glob": (str, type(None)),
    "content": (str,),
    "old_text": (str,),
    "new_text": (str,),
    "expected_hash": (str, type(None)),
    "offset": (int,),
    "limit": (int, type(None)),
    "command": (str,),
    "command_id": (str,),
    "timeout_seconds": (int, float, type(None)),
}


class SandboxProtocolError(Exception):
    """Raised when a framed worker message cannot be trusted."""


def encode_frame(value: Mapping[str, Any]) -> bytes:
    """Encode one bounded JSON object for the worker protocol."""
    try:
        payload = canonical_json_bytes(value)
    except (TypeError, UnicodeError, ValueError) as error:
        raise SandboxProtocolError("frame is not JSON") from error
    if len(payload) > MAX_FRAME_BYTES:
        raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE)
    return struct.pack(">I", len(payload)) + payload


def validate_request_params(
    method: str,
    params: Mapping[str, Any],
) -> None:
    """Validate the fixed fields and value types for one worker operation."""
    if not isinstance(method, str):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    schema = _REQUEST_FIELDS.get(method)
    if schema is None or not isinstance(params, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    required, optional = schema
    fields = set(params)
    if (
        any(not isinstance(field, str) for field in fields)
        or not required.issubset(fields)
        or fields - required - optional
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    for field in fields:
        value = params[field]
        accepted_types = _REQUEST_VALUE_TYPES[field]
        if isinstance(value, bool) and (int in accepted_types or float in accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(value, accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if field == "timeout_seconds" and isinstance(value, float):
            if not math.isfinite(value):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def validate_request_size(method: str, params: Mapping[str, Any]) -> None:
    """Apply the protocol input bound before a Local operation can mutate."""
    validate_request_params(method, params)
    try:
        encode_frame(
            {
                "version": PROTOCOL_VERSION,
                "request_id": _SIZE_CHECK_REQUEST_ID,
                "method": method,
                "params": dict(params),
            }
        )
    except AIError:
        raise
    except (SandboxProtocolError, UnicodeError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read and validate one length-prefixed JSON object."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise SandboxProtocolError("truncated frame header") from error
    size = struct.unpack(">I", header)[0]
    if size > MAX_FRAME_BYTES:
        raise SandboxProtocolError("frame exceeds protocol limit")
    try:
        payload = await reader.readexactly(size)
    except asyncio.IncompleteReadError as error:
        raise SandboxProtocolError("truncated frame") from error
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SandboxProtocolError("frame is not valid JSON") from error
    if not isinstance(value, dict):
        raise SandboxProtocolError("frame is not an object")
    return value


def protocol_error(code: ErrorCode, *, reason: str) -> dict[str, Any]:
    """Create the bounded error shape sent by the worker."""
    return {
        "code": code.value,
        "safe_details": {"reason": reason[:256]},
    }


__all__ = [
    "GUARDIAN_EXIT_CLEANUP_FAILED",
    "GUARDIAN_EXIT_OK",
    "GUARDIAN_EXIT_SESSION_FAILED",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "SandboxProtocolError",
    "WORKER_BUILD",
    "WORKER_EXIT_CLEANUP_FAILED",
    "WORKER_EXIT_OK",
    "WORKER_EXIT_SESSION_FAILED",
    "encode_frame",
    "protocol_error",
    "read_frame",
    "validate_request_params",
    "validate_request_size",
]
