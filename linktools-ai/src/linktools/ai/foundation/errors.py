#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stable error codes at the linktools.ai boundary."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Errors safe to expose at API, CLI, and workflow boundaries."""

    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_RELEASE_DISABLED = "AGENT_RELEASE_DISABLED"
    PROFILE_NOT_ALLOWED = "PROFILE_NOT_ALLOWED"
    CAPABILITY_DISABLED_FOR_PROFILE = "CAPABILITY_DISABLED_FOR_PROFILE"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    APPROVAL_NOT_PENDING = "APPROVAL_NOT_PENDING"
    APPROVAL_ALREADY_DECIDED = "APPROVAL_ALREADY_DECIDED"
    BUDGET_UNAVAILABLE = "BUDGET_UNAVAILABLE"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    EXECUTION_EXPIRED = "EXECUTION_EXPIRED"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    STORAGE_CONTEXT_REQUIRED = "STORAGE_CONTEXT_REQUIRED"
    STORAGE_CAPABILITY_MISSING = "STORAGE_CAPABILITY_MISSING"
    STORAGE_CONFLICT = "STORAGE_CONFLICT"
    STORAGE_NOT_FOUND = "STORAGE_NOT_FOUND"
    STORAGE_INTEGRITY_ERROR = "STORAGE_INTEGRITY_ERROR"
    RESULT_CONFLICT = "RESULT_CONFLICT"
    OUTPUT_CONTRACT_INVALID = "OUTPUT_CONTRACT_INVALID"
    INVALID_PAGE_TOKEN = "INVALID_PAGE_TOKEN"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    AGENT_BUNDLE_UNAVAILABLE = "AGENT_BUNDLE_UNAVAILABLE"
    ACTIVITY_SCOPE_REQUIRED = "ACTIVITY_SCOPE_REQUIRED"
    CONTEXT_PAYLOAD_TOO_LARGE = "CONTEXT_PAYLOAD_TOO_LARGE"
    TOOL_RESULT_UNKNOWN = "TOOL_RESULT_UNKNOWN"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    CAPABILITY_DEPENDENCY_MISSING = "CAPABILITY_DEPENDENCY_MISSING"
    CAPABILITY_CONFLICT = "CAPABILITY_CONFLICT"
    USAGE_LIMIT_EXCEEDED = "USAGE_LIMIT_EXCEEDED"
    MODEL_REQUEST_FAILED = "MODEL_REQUEST_FAILED"
    TOOL_FAILED = "TOOL_FAILED"
    WORKSPACE_RESTORE_FAILED = "WORKSPACE_RESTORE_FAILED"
    ARTIFACT_EXPORT_FAILED = "ARTIFACT_EXPORT_FAILED"
    BLOB_DELETE_IN_PROGRESS = "BLOB_DELETE_IN_PROGRESS"
    TENANT_CONTEXT_REQUIRED = "TENANT_CONTEXT_REQUIRED"
    UPSTREAM_VERSION_UNSUPPORTED = "UPSTREAM_VERSION_UNSUPPORTED"
    STORAGE_SCHEMA_INCOMPATIBLE = "STORAGE_SCHEMA_INCOMPATIBLE"
    STORAGE_TRANSACTION_FAILED = "STORAGE_TRANSACTION_FAILED"
    STORAGE_CURSOR_INVALID = "STORAGE_CURSOR_INVALID"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_BUSY = "SESSION_BUSY"
    SESSION_CONFLICT = "SESSION_CONFLICT"
    SESSION_CLEANUP_REQUIRED = "SESSION_CLEANUP_REQUIRED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_FENCE_STALE = "TASK_FENCE_STALE"
    TASK_DAG_INVALID = "TASK_DAG_INVALID"
    EVALUATION_INCOMPATIBLE = "EVALUATION_INCOMPATIBLE"
    TRACE_SEQUENCE_CONFLICT = "TRACE_SEQUENCE_CONFLICT"
    OUTPUT_SCHEMA_UNKNOWN = "OUTPUT_SCHEMA_UNKNOWN"
    OUTPUT_SCHEMA_DRIFT = "OUTPUT_SCHEMA_DRIFT"
    FEATURE_REGISTRY_FROZEN = "FEATURE_REGISTRY_FROZEN"
    LOCAL_PROJECT_NOT_FOUND = "LOCAL_PROJECT_NOT_FOUND"


class LinktoolsAIError(Exception):
    """Typed application error carrying a stable boundary code."""

    def __init__(self, code: ErrorCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


class StorageError(LinktoolsAIError):
    """Base error for generic storage primitives."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class StorageFeatureSupportError(StorageError):
    """Raised when a composition capability was not configured."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CAPABILITY_MISSING)


class StorageConflictError(StorageError):
    """Raised when a storage lease or compare-and-set loses a race."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CONFLICT)


class InvalidStoragePathError(StorageError):
    """Raised when a local storage path escapes its configured root."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_INTEGRITY_ERROR)


class StorageCorruptionError(StorageError):
    """Raised when persisted metadata references missing or invalid content."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_INTEGRITY_ERROR)


class AssetError(LinktoolsAIError):
    """Base error for asset parsing and persistence boundaries."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.STORAGE_UNAVAILABLE) -> None:
        super().__init__(code, message)


class AssetConflictError(AssetError):
    """Raised when an asset write conflicts with its content identity."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_CONFLICT)


class AssetNotFoundError(AssetError):
    """Raised when requested asset content does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STORAGE_NOT_FOUND)


class AssetParseError(AssetError):
    """Raised when asset content cannot be decoded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


class InvalidAssetError(AssetError):
    """Raised when a decoded asset violates its schema contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.OUTPUT_CONTRACT_INVALID)


__all__ = [
    "ErrorCode",
    "InvalidStoragePathError",
    "InvalidAssetError",
    "AssetConflictError",
    "AssetError",
    "AssetNotFoundError",
    "AssetParseError",
    "LinktoolsAIError",
    "StorageConflictError",
    "StorageCorruptionError",
    "StorageError",
    "StorageFeatureSupportError",
]
