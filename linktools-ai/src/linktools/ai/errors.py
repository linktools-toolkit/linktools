#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""errors.py: stable domain error hierarchy. Never identify an error by string
matching -- always by type (spec docs/linktools-ai.md section 32)."""


class LinktoolsAIError(Exception):
    """Base class for every error raised by linktools.ai."""


class ResourceError(LinktoolsAIError):
    """Base class for ResourceStore-related errors."""


class ResourceNotFoundError(ResourceError):
    pass


class ResourceConflictError(ResourceError):
    pass


class ResourcePreconditionFailedError(ResourceError):
    pass


class ResourceReadOnlyError(ResourceError):
    pass


class ResourceUnsupportedError(ResourceError):
    pass


class InvalidResourcePathError(ResourceError):
    pass


class StorageError(LinktoolsAIError):
    """Base class for Storage-facade-related errors."""


class StorageCapabilityError(StorageError):
    """Raised when an operation requires a StorageCapabilities flag the active
    Storage does not have (e.g. cross_store_transactions on FileStorage)."""


class IdempotencyConflictError(LinktoolsAIError):
    """Same idempotency key reused with a different request hash."""


class RunError(LinktoolsAIError):
    """Base class for Run-related errors."""


class RunNotFoundError(RunError):
    pass


class RunConflictError(RunError):
    pass


class RunCancelledError(RunError):
    pass


class InvalidRunTransitionError(RunError):
    pass


class SessionError(LinktoolsAIError):
    """Base class for Session-related errors."""


class EventError(LinktoolsAIError):
    """Base class for Event-related errors."""


class EventSequenceConflictError(EventError):
    pass


class ToolError(LinktoolsAIError):
    """Base class for Tool-execution-related errors."""


class ToolDeniedError(ToolError):
    pass


class ToolApprovalRequiredError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    pass


class PolicyError(LinktoolsAIError):
    """Base class for PolicyEngine-related errors."""


class ModelRoutingError(LinktoolsAIError):
    pass


class SwarmError(LinktoolsAIError):
    """Base class for Swarm-related errors."""


class SwarmRunNotFoundError(SwarmError):
    pass


class SwarmTaskNotFoundError(SwarmError):
    pass


class SwarmConflictError(SwarmError):
    pass


class InvalidSwarmTransitionError(SwarmError):
    pass


class SwarmLimitExceededError(SwarmError):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind
