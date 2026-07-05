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
