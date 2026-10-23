#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable errors exposed at the AI package boundary."""

import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from linktools.core import environ
from linktools.errors import Error

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

_SafeJsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["_SafeJsonValue"]
    | dict[str, "_SafeJsonValue"]
)
_DIAGNOSTIC_EXCEPTION_TYPE_MAX_LENGTH = 256
_DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_LENGTH = 2048


def _cause_digest(value: Mapping[str, str]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_json_mapping(
    value: "Mapping[str, _SafeJsonValue] | None",
) -> "dict[str, _SafeJsonValue]":
    if value is None:
        return {}
    normalized = _safe_json_value(value, set())
    if not isinstance(normalized, dict):
        raise TypeError("safe error details must be a mapping")
    return normalized


def _safe_json_value(value: object, seen: set[int]) -> _SafeJsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("safe error details require finite JSON numbers")
        return value
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            raise ValueError("safe error details cannot contain cycles")
        seen.add(identity)
        try:
            return [_safe_json_value(item, seen) for item in value]
        finally:
            seen.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("safe error details cannot contain cycles")
        seen.add(identity)
        try:
            normalized: dict[str, _SafeJsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("safe error detail keys must be strings")
                normalized[key] = _safe_json_value(item, seen)
            return normalized
        finally:
            seen.remove(identity)
    raise TypeError(
        f"safe error details contain unsupported value: {type(value).__name__}"
    )


class ErrorCode(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    ACTIVITY_SCOPE_REQUIRED = "ACTIVITY_SCOPE_REQUIRED"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    ASSET_CODEC_CONFLICT = "ASSET_CODEC_CONFLICT"
    ASSET_CODEC_UNKNOWN = "ASSET_CODEC_UNKNOWN"
    ASSET_LAYOUT_CONFLICT = "ASSET_LAYOUT_CONFLICT"
    ASSET_LAYOUT_UNKNOWN = "ASSET_LAYOUT_UNKNOWN"
    ASSET_CONFIG_TYPE_INVALID = "ASSET_CONFIG_TYPE_INVALID"
    ASSET_CONTENT_MISMATCH = "ASSET_CONTENT_MISMATCH"
    ASSET_CURSOR_INVALID = "ASSET_CURSOR_INVALID"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    CURSOR_INVALID = "CURSOR_INVALID"
    ASSET_ENV_MISSING = "ASSET_ENV_MISSING"
    ASSET_PATH_OUTSIDE_ROOT = "ASSET_PATH_OUTSIDE_ROOT"
    ASSET_PATH_ABSOLUTE = "ASSET_PATH_ABSOLUTE"
    ASSET_RECOVERY_REQUIRED = "ASSET_RECOVERY_REQUIRED"
    ASSET_BATCH_PARTIAL_FAILURE = "ASSET_BATCH_PARTIAL_FAILURE"
    APPROVAL_CONFLICT = "APPROVAL_CONFLICT"
    CONTEXT_PAYLOAD_TOO_LARGE = "CONTEXT_PAYLOAD_TOO_LARGE"
    EVALUATION_EVIDENCE_UNAVAILABLE = "EVALUATION_EVIDENCE_UNAVAILABLE"
    EVALUATION_INCOMPATIBLE = "EVALUATION_INCOMPATIBLE"
    CAPABILITY_CONFLICT = "CAPABILITY_CONFLICT"
    CAPABILITY_PROVIDER_UNKNOWN = "CAPABILITY_PROVIDER_UNKNOWN"
    CAPABILITY_REQUIRED_MISSING = "CAPABILITY_REQUIRED_MISSING"
    CAPABILITY_RESOLUTION_INVALID = "CAPABILITY_RESOLUTION_INVALID"
    CAPABILITY_POLICY_CONFLICT = "CAPABILITY_POLICY_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    OUTPUT_CONTRACT_INVALID = "OUTPUT_CONTRACT_INVALID"
    RUNTIME_DEPENDENCY_NOT_READY = "RUNTIME_DEPENDENCY_NOT_READY"
    SERVICE_NOT_READY = "SERVICE_NOT_READY"
    SESSION_BINDING_MISMATCH = "SESSION_BINDING_MISMATCH"
    SESSION_BUSY = "SESSION_BUSY"
    SESSION_CONFLICT = "SESSION_CONFLICT"
    SESSION_HISTORY_UNAVAILABLE = "SESSION_HISTORY_UNAVAILABLE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    STORAGE_BATCH_DUPLICATE_KEY = "STORAGE_BATCH_DUPLICATE_KEY"
    STORAGE_BATCH_PARTIAL_FAILURE = "STORAGE_BATCH_PARTIAL_FAILURE"
    STORAGE_CACHE_CORRUPT = "STORAGE_CACHE_CORRUPT"
    STORAGE_OWNER_MISMATCH = "STORAGE_OWNER_MISMATCH"
    STORAGE_LAYER_UNKNOWN = "STORAGE_LAYER_UNKNOWN"
    STORAGE_REVISION_NOTIFY_FAILED = "STORAGE_REVISION_NOTIFY_FAILED"
    STORAGE_READ_ONLY = "STORAGE_READ_ONLY"
    STORAGE_NOT_FOUND = "STORAGE_NOT_FOUND"
    STORAGE_PATH_INVALID = "STORAGE_PATH_INVALID"
    STORAGE_VERSION_UNSUPPORTED = "STORAGE_VERSION_UNSUPPORTED"
    STORAGE_CAPABILITY_MISSING = "STORAGE_CAPABILITY_MISSING"
    STORAGE_CONFLICT = "STORAGE_CONFLICT"
    STORAGE_INTEGRITY_ERROR = "STORAGE_INTEGRITY_ERROR"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    STORAGE_COMMIT_UNKNOWN = "STORAGE_COMMIT_UNKNOWN"
    STORAGE_ATOMIC_BATCH_UNSUPPORTED = "STORAGE_ATOMIC_BATCH_UNSUPPORTED"
    SNAPSHOT_CONFLICT = "SNAPSHOT_CONFLICT"
    SNAPSHOT_UNSUPPORTED = "SNAPSHOT_UNSUPPORTED"
    STORAGE_DEPENDENCY_NOT_READY = "STORAGE_DEPENDENCY_NOT_READY"
    OPTIONAL_DEPENDENCY_MISSING = "OPTIONAL_DEPENDENCY_MISSING"
    OUTPUT_SCHEMA_DRIFT = "OUTPUT_SCHEMA_DRIFT"
    OUTPUT_SCHEMA_UNKNOWN = "OUTPUT_SCHEMA_UNKNOWN"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    BINDING_NOT_REGISTERED = "BINDING_NOT_REGISTERED"
    BINDING_CONFLICT = "BINDING_CONFLICT"
    AGENT_BINDING_UNAVAILABLE = "AGENT_BINDING_UNAVAILABLE"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    SANDBOX_BUSY = "SANDBOX_BUSY"
    SANDBOX_SESSION_CLOSED = "SANDBOX_SESSION_CLOSED"
    SANDBOX_SESSION_LOST = "SANDBOX_SESSION_LOST"
    SANDBOX_CLEANUP_FAILED = "SANDBOX_CLEANUP_FAILED"
    TASK_DAG_INVALID = "TASK_DAG_INVALID"
    TASK_FENCE_STALE = "TASK_FENCE_STALE"
    TASK_OWNER_CONFLICT = "TASK_OWNER_CONFLICT"
    TASK_RESULT_CONFLICT = "TASK_RESULT_CONFLICT"
    TASK_TERMINAL_CONFLICT = "TASK_TERMINAL_CONFLICT"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_NODE_FAILED = "TASK_NODE_FAILED"
    TASK_DEPENDENCY_FAILED = "TASK_DEPENDENCY_FAILED"
    OBSERVER_FAILED = "OBSERVER_FAILED"
    OBSERVATION_FAILED = "OBSERVATION_FAILED"
    INPUT_CAPTURE_UNAVAILABLE = "INPUT_CAPTURE_UNAVAILABLE"
    INPUT_CONTEXT_UNAVAILABLE = "INPUT_CONTEXT_UNAVAILABLE"
    TASK_INPUT_PROJECTION_FAILED = "TASK_INPUT_PROJECTION_FAILED"
    WAIT_TIMEOUT = "WAIT_TIMEOUT"
    TASK_WAIT_TIMEOUT = "TASK_WAIT_TIMEOUT"
    HTTP_ROUTE_NOT_FOUND = "HTTP_ROUTE_NOT_FOUND"
    MODEL_REGISTRY_CONFLICT = "MODEL_REGISTRY_CONFLICT"
    MODEL_CONNECTION_NOT_FOUND = "MODEL_CONNECTION_NOT_FOUND"
    MODEL_CONNECTION_CONFLICT = "MODEL_CONNECTION_CONFLICT"
    MODEL_CONNECTION_UNSUPPORTED = "MODEL_CONNECTION_UNSUPPORTED"
    MODEL_CONFIG_INVALID = "MODEL_CONFIG_INVALID"
    MODEL_API_ERROR = "MODEL_API_ERROR"
    MODEL_REQUEST_REJECTED = "MODEL_REQUEST_REJECTED"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_RESPONSE_INVALID = "MODEL_RESPONSE_INVALID"
    MODEL_CONTENT_FILTERED = "MODEL_CONTENT_FILTERED"
    SESSION_ACTIVE_EXECUTIONS = "SESSION_ACTIVE_EXECUTIONS"
    SESSION_CLEANUP_REQUIRED = "SESSION_CLEANUP_REQUIRED"
    LOCAL_SKILL_CONFLICT = "LOCAL_SKILL_CONFLICT"
    SUBAGENT_CALL_LIMIT_EXCEEDED = "SUBAGENT_CALL_LIMIT_EXCEEDED"
    SUBAGENT_TIMEOUT = "SUBAGENT_TIMEOUT"
    REQUEST_FIELD_INVALID = "REQUEST_FIELD_INVALID"
    PROMPT_TOO_LARGE = "PROMPT_TOO_LARGE"
    EXTERNAL_RESULT_TOO_LARGE = "EXTERNAL_RESULT_TOO_LARGE"
    TOOL_ARGUMENTS_TOO_LARGE = "TOOL_ARGUMENTS_TOO_LARGE"
    OBSERVATION_PAYLOAD_TOO_LARGE = "OBSERVATION_PAYLOAD_TOO_LARGE"
    METRIC_NOT_FOUND = "METRIC_NOT_FOUND"
    METRIC_QUERY_LIMIT_EXCEEDED = "METRIC_QUERY_LIMIT_EXCEEDED"
    PAGE_LIMIT_INVALID = "PAGE_LIMIT_INVALID"
    IDEMPOTENCY_KEY_INVALID = "IDEMPOTENCY_KEY_INVALID"
    RUNTIME_SERVICE_MISMATCH = "RUNTIME_SERVICE_MISMATCH"
    STORAGE_CLOSED = "STORAGE_CLOSED"
    STORAGE_RECOVERY_REQUIRED = "STORAGE_RECOVERY_REQUIRED"
    SESSION_REVISION_CONFLICT = "SESSION_REVISION_CONFLICT"
    EXECUTION_RESULT_CONFLICT = "EXECUTION_RESULT_CONFLICT"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    EXECUTION_CONCURRENCY_LIMIT_EXCEEDED = "EXECUTION_CONCURRENCY_LIMIT_EXCEEDED"
    EXECUTION_NOT_READY = "EXECUTION_NOT_READY"
    EXECUTION_WAIT_TIMEOUT = "EXECUTION_WAIT_TIMEOUT"
    EXECUTION_START_UNKNOWN = "EXECUTION_START_UNKNOWN"
    EXECUTION_START_PERSISTENCE_FAILED = "EXECUTION_START_PERSISTENCE_FAILED"
    EXECUTION_HISTORY_UNAVAILABLE = "EXECUTION_HISTORY_UNAVAILABLE"
    EXECUTION_USAGE_LIMIT_EXCEEDED = "EXECUTION_USAGE_LIMIT_EXCEEDED"
    EXTERNAL_RESULT_CONFLICT = "EXTERNAL_RESULT_CONFLICT"
    AGENT_ID_INVALID = "AGENT_ID_INVALID"
    AGENT_INSTRUCTIONS_OUTSIDE_ROOT = "AGENT_INSTRUCTIONS_OUTSIDE_ROOT"
    TOOL_TIMEOUT_INVALID = "TOOL_TIMEOUT_INVALID"
    TOOL_OUTPUT_TRUNCATED = "TOOL_OUTPUT_TRUNCATED"
    TOOL_OPERATION_CONFLICT = "TOOL_OPERATION_CONFLICT"
    TOOL_EFFECT_UNKNOWN = "TOOL_EFFECT_UNKNOWN"
    TASK_EFFECT_UNKNOWN = "TASK_EFFECT_UNKNOWN"
    TOOL_RESULT_CONFLICT = "TOOL_RESULT_CONFLICT"
    TOOL_RETRY_REQUIRED = "TOOL_RETRY_REQUIRED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOO_MANY_PENDING_OPERATIONS = "TOO_MANY_PENDING_OPERATIONS"
    TASK_GRAPH_CYCLE = "TASK_GRAPH_CYCLE"
    TASK_GRAPH_DEADLOCK = "TASK_GRAPH_DEADLOCK"
    TASK_DEPENDENCY_UNKNOWN = "TASK_DEPENDENCY_UNKNOWN"
    OUTPUT_SCHEMA_REVISION_REQUIRED = "OUTPUT_SCHEMA_REVISION_REQUIRED"
    ASSET_VERSION_NOT_FOUND = "ASSET_VERSION_NOT_FOUND"
    ASSET_VERSION_LAYER_UNKNOWN = "ASSET_VERSION_LAYER_UNKNOWN"
    REDACTION_FAILED = "REDACTION_FAILED"
    MCP_CONNECTION_FAILED = "MCP_CONNECTION_FAILED"
    MCP_CLEANUP_FAILED = "MCP_CLEANUP_FAILED"
    MCP_RESPONSE_TOO_LARGE = "MCP_RESPONSE_TOO_LARGE"
    LOCAL_SHELL_PLATFORM_UNSUPPORTED = "LOCAL_SHELL_PLATFORM_UNSUPPORTED"
    WORKSPACE_CONFIG_INVALID = "WORKSPACE_CONFIG_INVALID"


@dataclass(frozen=True, slots=True)
class ErrorDiagnostics:
    """Failure detail captured before wrapping, without frame locals.

    The original three fields remain the bounded summary. ``exception_chain``
    retains full messages along the visible cause/context chain, outermost
    first; ``traceback`` also includes exception-group branches and notes.
    Missing optional values mean they were not collected, not an empty failure.
    With ``content_omission_reason``, messages and notes are never inspected;
    the chain retains exception types, relationships and group paths while the
    traceback retains frames without locals or source lines.
    """

    exception_type: str
    exception_message: str
    cause_digest: str
    exception_chain: tuple[Mapping[str, str], ...] | None = None
    traceback: str | None = None
    content_omission_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.exception_type, str)
            or not self.exception_type
            or len(self.exception_type) > _DIAGNOSTIC_EXCEPTION_TYPE_MAX_LENGTH
        ):
            raise ValueError("diagnostic exception type is invalid")
        if (
            not isinstance(self.exception_message, str)
            or len(self.exception_message) > _DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_LENGTH
        ):
            raise ValueError("diagnostic exception message is invalid")
        if not _is_sha256(self.cause_digest):
            raise ValueError("diagnostic cause digest must be lowercase SHA-256")
        if self.traceback is not None and not isinstance(self.traceback, str):
            raise TypeError("diagnostic traceback must be text or null")
        if self.content_omission_reason is not None:
            if not isinstance(self.content_omission_reason, str) or not self.content_omission_reason.strip():
                raise ValueError("diagnostic content omission reason must be a nonempty string or null")
            object.__setattr__(self, "content_omission_reason", environ.logging.redact(self.content_omission_reason))
        if self.exception_chain is not None:
            if not isinstance(self.exception_chain, tuple):
                raise TypeError("diagnostic exception chain must be a tuple or null")
            chain = []
            for cause in self.exception_chain:
                if (
                    not isinstance(cause, Mapping)
                    or not {"exception_type", "exception_message", "relation"}.issubset(cause)
                    or not all(isinstance(name, str) for name in cause)
                    or not all(isinstance(value, str) for value in cause.values())
                ):
                    raise ValueError("diagnostic exception chain entry is invalid")
                chain.append({
                    name: environ.logging.redact(value) for name, value in cause.items()
                })
            object.__setattr__(self, "exception_chain", tuple(chain))
        object.__setattr__(self, "exception_type", environ.logging.redact(self.exception_type))
        object.__setattr__(self, "exception_message", environ.logging.redact(self.exception_message))
        if self.traceback is not None:
            object.__setattr__(self, "traceback", environ.logging.redact(self.traceback))

    def to_payload(self) -> "dict[str, _SafeJsonValue]":
        """Return all collected diagnostics as JSON, without display limits."""
        payload: dict[str, _SafeJsonValue] = {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "cause_digest": self.cause_digest,
        }
        if self.exception_chain is not None:
            payload["exception_chain"] = [dict(cause) for cause in self.exception_chain]
        if self.traceback is not None:
            payload["traceback"] = self.traceback
        if self.content_omission_reason is not None:
            payload["content_omission_reason"] = self.content_omission_reason
        return payload

    @classmethod
    def from_exception(
        cls, error: BaseException, *, content_omission_reason: "str | None" = None,
    ) -> "ErrorDiagnostics":
        """Capture a failure, optionally omitting all original exception content.

        An omission reason is a diagnostic label, not request content. Omission
        reads only exception types, relationships and traceback frame locations;
        it does not format original messages, notes, source lines or frame locals.
        """
        if content_omission_reason is not None:
            if not isinstance(content_omission_reason, str) or not content_omission_reason.strip():
                raise ValueError("diagnostic content omission reason must be a nonempty string or null")
            return cls._from_exception_omitting_content(error, content_omission_reason)
        exception_type = type(error).__name__
        chain: list[Mapping[str, str]] = []
        current: BaseException | None = error
        seen: set[int] = set()
        relation = "root"
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            try:
                message = str(current)
            except BaseException:
                message = ""
            chain.append({
                "exception_type": type(current).__name__,
                "exception_message": message,
                "relation": relation,
            })
            if current.__cause__ is not None:
                current, relation = current.__cause__, "cause"
            elif not current.__suppress_context__:
                current, relation = current.__context__, "context"
            else:
                current = None
        exception_message = chain[0]["exception_message"]
        # Python's default group formatting and sys.tracebacklimit are display
        # limits, not bounds for a durable failure diagnostic.
        group_options = (
            {"max_group_width": sys.maxsize, "max_group_depth": sys.maxsize}
            if sys.version_info >= (3, 11) else {}
        )
        try:
            formatted = traceback.TracebackException(
                type(error), error, error.__traceback__, limit=sys.maxsize,
                capture_locals=False, **group_options,
            )
            stack = "".join(formatted.format(chain=True))
        except BaseException:
            # Diagnostic formatting must not replace the failure being handled.
            stack = None
        return cls(
            environ.logging.redact(exception_type)[:_DIAGNOSTIC_EXCEPTION_TYPE_MAX_LENGTH],
            environ.logging.redact(exception_message)[:_DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_LENGTH],
            _cause_digest(
                {
                    "exception_message": exception_message,
                    "exception_type": exception_type,
                }
            ),
            exception_chain=tuple(chain),
            traceback=stack,
        )

    @classmethod
    def _from_exception_omitting_content(cls, error: BaseException, reason: str) -> "ErrorDiagnostics":
        reason = environ.logging.redact(reason)
        message = f"[content omitted: {reason}]"
        chain: list[Mapping[str, str]] = []
        stack = ["Exception traceback (original messages, notes and source lines omitted):\n"]
        seen: dict[int, str] = {}
        pending = [("root", "root", error)]
        while pending:
            path, relation, current = pending.pop()
            exception_type = type(current).__name__
            previous = seen.get(id(current))
            entry = {
                "exception_type": exception_type,
                "exception_message": message,
                "relation": relation,
                "path": path,
            }
            if previous is not None:
                entry["same_exception_as"] = previous
            chain.append(entry)
            alias = f" (same exception as {previous})" if previous is not None else ""
            stack.append(f"{path}{alias}: {exception_type}: {message}\n")
            # Reading code locations directly avoids traceback formatters that
            # stringify exceptions, inspect notes or load source text.
            for frame, lineno in traceback.walk_tb(current.__traceback__):
                code = frame.f_code
                stack.append(f'  File "{code.co_filename}", line {lineno}, in {code.co_name}\n')
            if previous is not None:
                continue
            seen[id(current)] = path
            if isinstance(current, BaseExceptionGroup):
                pending.extend(
                    (f"{path}.branch[{index}]", "branch", child)
                    for index, child in reversed(list(enumerate(current.exceptions)))
                )
            if current.__cause__ is not None:
                pending.append(("cause" if path == "root" else f"{path}.cause", "cause", current.__cause__))
            elif not current.__suppress_context__ and current.__context__ is not None:
                pending.append(("context" if path == "root" else f"{path}.context", "context", current.__context__))
        stack_text = environ.logging.redact("".join(stack))
        return cls(
            environ.logging.redact(type(error).__name__)[:_DIAGNOSTIC_EXCEPTION_TYPE_MAX_LENGTH],
            message[:_DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_LENGTH],
            _cause_digest({
                "exception_type": environ.logging.redact(type(error).__name__),
                "exception_message": message,
                "traceback": stack_text,
            }),
            exception_chain=tuple(chain),
            traceback=stack_text,
            content_omission_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class SafeError:
    code: str
    category: str
    retryable: bool
    operation_id: str
    safe_details: "Mapping[str, _SafeJsonValue]"
    cause_digest: str


class AIError(Error):
    """An error with a stable machine-readable code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        category: "str | None" = None,
        retryable: "bool | None" = None,
        operation_id: "str | None" = None,
        safe_details: "Mapping[str, _SafeJsonValue] | None" = None,
        diagnostics: "ErrorDiagnostics | None" = None,
    ) -> None:
        super().__init__(message or code.value)
        if diagnostics is not None and not isinstance(diagnostics, ErrorDiagnostics):
            raise TypeError("error diagnostics are invalid")
        self.code = code
        self.category = category or code.value.split("_", 1)[0]
        self.retryable = (
            code
            in {
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                ErrorCode.SERVICE_NOT_READY,
                ErrorCode.STORAGE_CACHE_CORRUPT,
                ErrorCode.STORAGE_OWNER_MISMATCH,
                ErrorCode.STORAGE_UNAVAILABLE,
                ErrorCode.STORAGE_DEPENDENCY_NOT_READY,
                ErrorCode.SANDBOX_BUSY,
                ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                ErrorCode.SESSION_CLEANUP_REQUIRED,
                ErrorCode.MODEL_RATE_LIMITED,
                ErrorCode.MODEL_TIMEOUT,
                ErrorCode.MODEL_UNAVAILABLE,
                ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED,
                ErrorCode.EXECUTION_NOT_READY,
                ErrorCode.EXECUTION_WAIT_TIMEOUT,
                ErrorCode.TASK_WAIT_TIMEOUT,
                ErrorCode.WAIT_TIMEOUT,
            }
            if retryable is None
            else retryable
        )
        self.operation_id = operation_id
        self.safe_details = _safe_json_mapping(safe_details)
        self.diagnostics = diagnostics

    def to_safe_error(self, *, operation_id: str) -> SafeError:
        return SafeError(
            self.code.value,
            self.category,
            self.retryable,
            operation_id,
            self.safe_details,
            _cause_digest({"type": type(self).__name__, "code": self.code.value}),
        )


class StorageError(AIError):
    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class InvalidStoragePathError(StorageError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_PATH_INVALID)


class AssetError(AIError):
    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class ObservationError(AIError):
    """An observer or optional presentation stream failed without changing durable state.

    Callback failures preserve the original cause and last acknowledged cursor.
    Stream failures exclude authoritative reads, durable decoding and protocol
    validation errors. Cancellation is never converted to this error. A stream
    failure during cleanup may indicate unfinished local work and is not a
    successful observation result.
    """

    def __init__(
        self,
        origin: Literal["callback", "stream"],
        *,
        cursor: str | None,
        cause_code: str | None = None,
        safe_details: "Mapping[str, _SafeJsonValue] | None" = None,
        diagnostics: "ErrorDiagnostics | None" = None,
    ) -> None:
        if origin not in {"callback", "stream"}:
            raise ValueError("observation origin is invalid")
        if cursor is not None and not isinstance(cursor, str):
            raise TypeError("observation cursor is invalid")
        if cause_code is not None and (not isinstance(cause_code, str) or not cause_code):
            raise ValueError("observation cause code is invalid")
        code = (
            ErrorCode.OBSERVER_FAILED
            if origin == "callback"
            else ErrorCode.OBSERVATION_FAILED
        )
        super().__init__(
            code,
            safe_details=safe_details,
            diagnostics=diagnostics,
        )
        self.origin = origin
        self.cursor = cursor
        self.cause_code = cause_code


__all__ = [
    "AIError",
    "AssetError",
    "ErrorCode",
    "ErrorDiagnostics",
    "InvalidStoragePathError",
    "SafeError",
    "StorageError",
    "ObservationError",
]
