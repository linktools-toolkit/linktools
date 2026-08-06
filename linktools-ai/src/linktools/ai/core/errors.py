#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable errors exposed at package boundaries."""

from enum import StrEnum
from collections.abc import Mapping

from .json import JsonValue


class ErrorCode(StrEnum):
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    ACTIVITY_SCOPE_REQUIRED = "ACTIVITY_SCOPE_REQUIRED"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    ASSET_CODEC_CONFLICT = "ASSET_CODEC_CONFLICT"
    ASSET_CODEC_UNKNOWN = "ASSET_CODEC_UNKNOWN"
    ASSET_CONFIG_TYPE_INVALID = "ASSET_CONFIG_TYPE_INVALID"
    ASSET_CONTENT_MISMATCH = "ASSET_CONTENT_MISMATCH"
    ASSET_CURSOR_INVALID = "ASSET_CURSOR_INVALID"
    ASSET_ENV_MISSING = "ASSET_ENV_MISSING"
    ASSET_REVISION_CONFLICT = "ASSET_REVISION_CONFLICT"
    ASSET_PATH_OUTSIDE_ROOT = "ASSET_PATH_OUTSIDE_ROOT"
    ASSET_PATH_ABSOLUTE = "ASSET_PATH_ABSOLUTE"
    ASSET_RECOVERY_REQUIRED = "ASSET_RECOVERY_REQUIRED"
    ASSET_BATCH_PARTIAL_FAILURE = "ASSET_BATCH_PARTIAL_FAILURE"
    APPROVAL_CONFLICT = "APPROVAL_CONFLICT"
    CAPABILITY_DISABLED_FOR_PROFILE = "CAPABILITY_DISABLED_FOR_PROFILE"
    CONTEXT_PAYLOAD_TOO_LARGE = "CONTEXT_PAYLOAD_TOO_LARGE"
    EVALUATION_INCOMPATIBLE = "EVALUATION_INCOMPATIBLE"
    FEATURE_REGISTRY_FROZEN = "FEATURE_REGISTRY_FROZEN"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    OUTPUT_CONTRACT_INVALID = "OUTPUT_CONTRACT_INVALID"
    PROFILE_NOT_ALLOWED = "PROFILE_NOT_ALLOWED"
    RUNTIME_DEPENDENCY_NOT_READY = "RUNTIME_DEPENDENCY_NOT_READY"
    SERVICE_NOT_READY = "SERVICE_NOT_READY"
    SESSION_BINDING_MISMATCH = "SESSION_BINDING_MISMATCH"
    SESSION_BUSY = "SESSION_BUSY"
    SESSION_CONFLICT = "SESSION_CONFLICT"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    STORAGE_BATCH_DUPLICATE_KEY = "STORAGE_BATCH_DUPLICATE_KEY"
    STORAGE_BATCH_PARTIAL_FAILURE = "STORAGE_BATCH_PARTIAL_FAILURE"
    STORAGE_CACHE_CORRUPT = "STORAGE_CACHE_CORRUPT"
    STORAGE_OWNER_MISMATCH = "STORAGE_OWNER_MISMATCH"
    STORAGE_REVISION_NOTIFY_FAILED = "STORAGE_REVISION_NOTIFY_FAILED"
    STORAGE_READ_ONLY = "STORAGE_READ_ONLY"
    STORAGE_NOT_FOUND = "STORAGE_NOT_FOUND"
    STORAGE_VERSION_UNSUPPORTED = "STORAGE_VERSION_UNSUPPORTED"
    STORAGE_CAPABILITY_MISSING = "STORAGE_CAPABILITY_MISSING"
    STORAGE_CONFLICT = "STORAGE_CONFLICT"
    STORAGE_INTEGRITY_ERROR = "STORAGE_INTEGRITY_ERROR"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    OPTIONAL_DEPENDENCY_MISSING = "OPTIONAL_DEPENDENCY_MISSING"
    OUTPUT_SCHEMA_DRIFT = "OUTPUT_SCHEMA_DRIFT"
    OUTPUT_SCHEMA_UNKNOWN = "OUTPUT_SCHEMA_UNKNOWN"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    TASK_DAG_INVALID = "TASK_DAG_INVALID"
    TASK_FENCE_STALE = "TASK_FENCE_STALE"
    TASK_OWNER_CONFLICT = "TASK_OWNER_CONFLICT"
    TASK_RESULT_CONFLICT = "TASK_RESULT_CONFLICT"
    TASK_TERMINAL_CONFLICT = "TASK_TERMINAL_CONFLICT"
    TASK_NOT_READY = "TASK_NOT_READY"
    TRACE_SEQUENCE_CONFLICT = "TRACE_SEQUENCE_CONFLICT"
    LINKTOOLS_AI_RELEASE_MISMATCH = "LINKTOOLS_AI_RELEASE_MISMATCH"
    HTTP_ROUTE_NOT_FOUND = "HTTP_ROUTE_NOT_FOUND"
    MIDDLEWARE_FAILED = "MIDDLEWARE_FAILED"
    MODEL_REGISTRY_CONFLICT = "MODEL_REGISTRY_CONFLICT"
    SESSION_ACTIVE_EXECUTIONS = "SESSION_ACTIVE_EXECUTIONS"
    SESSION_CLEANUP_REQUIRED = "SESSION_CLEANUP_REQUIRED"
    LOCAL_SKILL_CONFLICT = "LOCAL_SKILL_CONFLICT"


class LinktoolsAIError(Exception):
    """An error with a stable machine-readable code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        category: 'str | None' = None,
        retryable: 'bool | None' = None,
        operation_id: 'str | None' = None,
        safe_details: 'Mapping[str, JsonValue] | None' = None,
    ) -> None:
        super().__init__(message or code.value)
        self.code = code
        self.category = category or code.value.split("_", 1)[0]
        self.retryable = code in {
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            ErrorCode.SERVICE_NOT_READY,
            ErrorCode.STORAGE_CACHE_CORRUPT,
            ErrorCode.STORAGE_OWNER_MISMATCH,
            ErrorCode.STORAGE_UNAVAILABLE,
            ErrorCode.SESSION_ACTIVE_EXECUTIONS,
            ErrorCode.SESSION_CLEANUP_REQUIRED,
        } if retryable is None else retryable
        self.operation_id = operation_id
        self.safe_details = dict(safe_details or {})


class StorageError(LinktoolsAIError):
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
        super().__init__(message, ErrorCode.STORAGE_INTEGRITY_ERROR)


class AssetError(LinktoolsAIError):
    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class AssetConflictError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CONFLICT)


class AssetNotFoundError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_NOT_FOUND)


class AssetParseError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


class InvalidAssetError(AssetError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


__all__ = [
    "AssetConflictError",
    "AssetError",
    "AssetNotFoundError",
    "AssetParseError",
    "ErrorCode",
    "InvalidAssetError",
    "InvalidStoragePathError",
    "LinktoolsAIError",
    "StorageConflictError",
    "StorageCorruptionError",
    "StorageError",
]
