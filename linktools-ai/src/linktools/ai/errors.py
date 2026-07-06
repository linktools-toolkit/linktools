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


class RunPaused(RunError):
    """Raised by ToolExecutor (under ``pause_on_approval=True``) and propagated
    through pydantic-ai's tool-execution stack out to AgentRunner, which
    checkpoints state, transitions the Run to WAITING_APPROVAL, and stops.

    This is a control-flow signal, NOT an error condition -- it's a RunError
    (not a ToolError) precisely so PolicyCapability.before_tool_execute (which
    only catches ToolDeniedError/ToolApprovalRequiredError -> SkipToolExecution)
    lets it propagate. AgentRunner catches it; nothing else should.

    Carries both ids AgentRunner needs: ``run_id`` (already resolved through
    ToolExecutor.run_id_resolver, so it matches the persisted ApprovalRequest)
    and ``approval_id`` (the PENDING request the human must approve/reject)."""

    def __init__(self, run_id: str, approval_id: str) -> None:
        super().__init__(
            f"run paused waiting for approval: run_id={run_id} approval_id={approval_id}"
        )
        self.run_id = run_id
        self.approval_id = approval_id


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


class ToolIdempotencyConflictError(ToolError):
    pass


class IdempotencyInProgressError(ToolError):
    """Raised by ToolExecutor when an idempotent call hits a RESERVED record
    (another in-flight call owns the reservation). §11.2 lists "wait / return
    in-progress / reject duplicate" as policy choices; for now the executor
    rejects -- the caller can retry once the in-flight call completes and the
    record moves to COMPLETED or FAILED."""



class PolicyError(LinktoolsAIError):
    """Base class for PolicyEngine-related errors."""


class ModelRoutingError(LinktoolsAIError):
    pass


class ModelPolicyExceededError(LinktoolsAIError):
    """Raised when a ModelPolicy limit (max_tokens, ...) is violated by a model
    call's actual usage. Carries ``kind`` so callers can distinguish which limit
    fired (currently only ``"max_tokens"``; ``"budget"`` is deferred until
    cost-per-token rates exist)."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class SwarmError(LinktoolsAIError):
    """Base class for Swarm-related errors."""


class SwarmRunNotFoundError(SwarmError):
    pass


class SwarmTaskNotFoundError(SwarmError):
    pass


class SwarmTaskConflictError(SwarmError):
    pass


class SwarmConflictError(SwarmError):
    pass


class InvalidSwarmTransitionError(SwarmError):
    pass


class SwarmLimitExceededError(SwarmError):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class MemoryError(LinktoolsAIError):
    """Base class for Memory-related errors."""


class MemoryNotFoundError(MemoryError):
    pass


class MemoryConflictError(MemoryError):
    pass


class RegistryError(LinktoolsAIError):
    """Base class for spec-registry errors (loading/parsing spec files)."""


class RegistryNotFoundError(RegistryError):
    pass


class RegistryConflictError(RegistryError):
    pass


class RegistryParseError(RegistryError):
    pass


class InvalidSpecError(RegistryError):
    """A parsed spec is structurally present but semantically invalid."""


class ApprovalError(LinktoolsAIError):
    """Base class for Approval-store errors."""


class ApprovalNotFoundError(ApprovalError):
    pass


class ApprovalConflictError(ApprovalError):
    pass


class InvalidApprovalTransitionError(ApprovalError):
    pass
