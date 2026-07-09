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


class StorageTransactionNotSupportedError(StorageCapabilityError):
    """cross_store_transactions is False on this Storage but a caller requested
    an atomic cross-store write."""


class StorageConcurrencyNotSupportedError(StorageCapabilityError):
    """optimistic_concurrency is False but a caller requested CAS-style updates."""


class StorageLeaseNotSupportedError(StorageCapabilityError):
    """leasing is False but a caller (e.g. swarm claim) requested a lease."""


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
    persists the ApprovalRequest, checkpoints state, transitions the Run to
    WAITING_APPROVAL, and appends the pause events -- all atomically in one
    UnitOfWork on SqlAlchemy storage (review3 §5, Package A / P0-6).

    This is a control-flow signal, NOT an error condition -- it's a RunError
    (not a ToolError) precisely so PolicyCapability.before_tool_execute (which
    only catches ToolDeniedError/ToolApprovalRequiredError -> SkipToolExecution)
    lets it propagate. AgentRunner catches it; nothing else should.

    ``approval_id`` is a fresh id MINTED here (not yet persisted anywhere) --
    ToolExecutor no longer writes the ApprovalRequest itself; it only mints
    the id so the id it reports is the same one AgentRunner's suspension
    handler will actually persist. ``run_id`` is already resolved through
    ToolExecutor.run_id_resolver. The remaining fields carry everything the
    suspension handler needs to construct and persist the ApprovalRequest
    without ToolExecutor touching the ApprovalStore. Only primitive types are
    used here (no domain dataclass import) to keep this module dependency-free."""

    def __init__(
        self, run_id: str, approval_id: str, *,
        tool_call_id: "str | None" = None,
        tool_name: "str | None" = None,
        reason: "str | None" = None,
        arguments: "dict | None" = None,
        idempotency_key: "str | None" = None,
    ) -> None:
        super().__init__(
            f"run paused waiting for approval: run_id={run_id} approval_id={approval_id}"
        )
        self.run_id = run_id
        self.approval_id = approval_id
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.reason = reason
        self.arguments = arguments or {}
        self.idempotency_key = idempotency_key


class SessionError(LinktoolsAIError):
    """Base class for Session-related errors."""


class SessionSequenceConflictError(SessionError):
    """Raised when the SessionStore cannot reserve a unique message sequence
    after repeated conflicts (G6/review3 §6: the store is the sole sequence
    authority, mirroring EventSequenceConflictError)."""


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


# --- Capability resolution tree -----------------------------------------
# Resolving AgentSpec.tools into concrete capability bundles can fail in two
# qualitatively different ways: a referenced capability cannot be found, or two
# capabilities collide. Both carry agent_id / ref so callers can pinpoint the
# failing declaration instead of grepping strings.

class CapabilityResolutionError(LinktoolsAIError):
    """Base class for capability-resolution failures (assemble-time)."""


class CapabilityNotFoundError(CapabilityResolutionError):
    pass


class CapabilityConflictError(CapabilityResolutionError):
    """Two capabilities produced the same tool name; resolution never silently
    overwrites."""


class SkillNotFoundError(CapabilityNotFoundError):
    pass


class MCPServerNotFoundError(CapabilityNotFoundError):
    pass


class MCPConnectionError(LinktoolsAIError):
    """An MCP server connection could not be established or was lost."""


class MCPToolError(LinktoolsAIError):
    """An MCP tool invocation failed at the protocol/transport layer."""


class PackageNotFoundError(CapabilityNotFoundError):
    pass


class PackageResourceNotFoundError(CapabilityNotFoundError):
    pass


class PackageResourceAccessDeniedError(PolicyError):
    """A package resource path was outside the allowed scope/extension set."""


class PackageEntrypointNotFoundError(CapabilityNotFoundError):
    pass


class PackageEntrypointDeniedError(PolicyError):
    """An entrypoint kind/name was not on the declared allowlist."""


class SubagentNotFoundError(CapabilityNotFoundError):
    pass


class SubagentDepthExceededError(PolicyError):
    """A subagent call would exceed the configured max_depth."""

    def __init__(self, message: str, *, depth: int, max_depth: int) -> None:
        super().__init__(message)
        self.depth = depth
        self.max_depth = max_depth


class SubagentExecutionError(LinktoolsAIError):
    """A delegated subagent run failed; carries the structured child error."""

    def __init__(self, message: str, *, error: "dict | None" = None) -> None:
        super().__init__(message)
        self.error = error


class ModelOutputValidationError(LinktoolsAIError):
    """A model response could not be validated against the expected output."""


class ModelTurnLimitExceededError(ModelPolicyExceededError):
    """A run exhausted its turn/request budget. (Stable alias of the
    model-registry ModelTurnLimitExceeded; identify by type, not string.)"""


class ApprovalError(LinktoolsAIError):
    """Base class for Approval-store errors."""


class ApprovalNotFoundError(ApprovalError):
    pass


class ApprovalConflictError(ApprovalError):
    pass


class InvalidApprovalTransitionError(ApprovalError):
    pass
