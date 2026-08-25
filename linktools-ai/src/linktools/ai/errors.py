#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable errors exposed at the AI package boundary."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from linktools.errors import Error

_SafeJsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["_SafeJsonValue"]
    | dict[str, "_SafeJsonValue"]
)


def _cause_digest(value: Mapping[str, str]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    EVALUATION_INCOMPATIBLE = "EVALUATION_INCOMPATIBLE"
    CAPABILITY_CONFLICT = "CAPABILITY_CONFLICT"
    CAPABILITY_FINGERPRINT_INVALID = "CAPABILITY_FINGERPRINT_INVALID"
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
    STORAGE_DEPENDENCY_NOT_READY = "STORAGE_DEPENDENCY_NOT_READY"
    OPTIONAL_DEPENDENCY_MISSING = "OPTIONAL_DEPENDENCY_MISSING"
    OUTPUT_SCHEMA_DRIFT = "OUTPUT_SCHEMA_DRIFT"
    OUTPUT_SCHEMA_UNKNOWN = "OUTPUT_SCHEMA_UNKNOWN"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    BINDING_NOT_REGISTERED = "BINDING_NOT_REGISTERED"
    BINDING_CONFLICT = "BINDING_CONFLICT"
    AGENT_DEFINITION_UNAVAILABLE = "AGENT_DEFINITION_UNAVAILABLE"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    TASK_DAG_INVALID = "TASK_DAG_INVALID"
    TASK_FENCE_STALE = "TASK_FENCE_STALE"
    TASK_OWNER_CONFLICT = "TASK_OWNER_CONFLICT"
    TASK_RESULT_CONFLICT = "TASK_RESULT_CONFLICT"
    TASK_TERMINAL_CONFLICT = "TASK_TERMINAL_CONFLICT"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_NODE_FAILED = "TASK_NODE_FAILED"
    TASK_DEPENDENCY_FAILED = "TASK_DEPENDENCY_FAILED"
    TASK_WAIT_TIMEOUT = "TASK_WAIT_TIMEOUT"
    LINKTOOLS_AI_RELEASE_MISMATCH = "LINKTOOLS_AI_RELEASE_MISMATCH"
    HTTP_ROUTE_NOT_FOUND = "HTTP_ROUTE_NOT_FOUND"
    MIDDLEWARE_FAILED = "MIDDLEWARE_FAILED"
    MODEL_REGISTRY_CONFLICT = "MODEL_REGISTRY_CONFLICT"
    MODEL_CONNECTION_NOT_FOUND = "MODEL_CONNECTION_NOT_FOUND"
    MODEL_CONNECTION_CONFLICT = "MODEL_CONNECTION_CONFLICT"
    MODEL_CONNECTION_UNSUPPORTED = "MODEL_CONNECTION_UNSUPPORTED"
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
    REQUEST_FIELD_INVALID = "REQUEST_FIELD_INVALID"
    PROMPT_TOO_LARGE = "PROMPT_TOO_LARGE"
    EXTERNAL_RESULT_TOO_LARGE = "EXTERNAL_RESULT_TOO_LARGE"
    TOOL_ARGUMENTS_TOO_LARGE = "TOOL_ARGUMENTS_TOO_LARGE"
    OBSERVATION_PAYLOAD_TOO_LARGE = "OBSERVATION_PAYLOAD_TOO_LARGE"
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
    TOOL_RESULT_CONFLICT = "TOOL_RESULT_CONFLICT"
    TOOL_RETRY_REQUIRED = "TOOL_RETRY_REQUIRED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOO_MANY_PENDING_OPERATIONS = "TOO_MANY_PENDING_OPERATIONS"
    TASK_GRAPH_CYCLE = "TASK_GRAPH_CYCLE"
    TASK_GRAPH_DEADLOCK = "TASK_GRAPH_DEADLOCK"
    TASK_DEPENDENCY_UNKNOWN = "TASK_DEPENDENCY_UNKNOWN"
    OUTPUT_SCHEMA_REVISION_REQUIRED = "OUTPUT_SCHEMA_REVISION_REQUIRED"
    ASSET_VERSION_NOT_FOUND = "ASSET_VERSION_NOT_FOUND"
    ASSET_VERSION_OWNER_UNKNOWN = "ASSET_VERSION_OWNER_UNKNOWN"
    REDACTION_FAILED = "REDACTION_FAILED"
    MCP_RESPONSE_TOO_LARGE = "MCP_RESPONSE_TOO_LARGE"
    LOCAL_SHELL_PLATFORM_UNSUPPORTED = "LOCAL_SHELL_PLATFORM_UNSUPPORTED"


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
    ) -> None:
        super().__init__(message or code.value)
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
                ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                ErrorCode.SESSION_CLEANUP_REQUIRED,
                ErrorCode.MODEL_RATE_LIMITED,
                ErrorCode.MODEL_TIMEOUT,
                ErrorCode.MODEL_UNAVAILABLE,
                ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED,
                ErrorCode.EXECUTION_NOT_READY,
                ErrorCode.EXECUTION_WAIT_TIMEOUT,
                ErrorCode.TASK_WAIT_TIMEOUT,
            }
            if retryable is None
            else retryable
        )
        self.operation_id = operation_id
        self.safe_details = _safe_json_mapping(safe_details)

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


class StorageConflictError(StorageError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CONFLICT)


class StorageCorruptionError(StorageError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_INTEGRITY_ERROR)


class InvalidStoragePathError(StorageError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_PATH_INVALID)


class AssetError(AIError):
    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class AssetConflictError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CONFLICT)


class AssetNotFoundError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.ASSET_NOT_FOUND)


class AssetParseError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


class InvalidAssetError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


__all__ = [
    "AIError",
    "AssetConflictError",
    "AssetError",
    "AssetNotFoundError",
    "AssetParseError",
    "ErrorCode",
    "InvalidAssetError",
    "InvalidStoragePathError",
    "SafeError",
    "StorageConflictError",
    "StorageCorruptionError",
    "StorageError",
]
