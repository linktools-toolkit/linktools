#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""errors.py: stable domain error hierarchy. Never identify an error by string
matching -- always by type."""

from enum import Enum


class LinktoolsAIError(Exception):
    """Base class for every error raised by linktools.ai."""


class RuntimeInitializationError(LinktoolsAIError):
    """The runtime cannot safely initialize a required component."""


class UnsafeExecutionBackendError(RuntimeInitializationError):
    """A trusted-local backend was used where tenant isolation is required."""


class SecurityError(LinktoolsAIError):
    """Security-domain failures: a sensitive operation was attempted without a
    valid Principal, or the Principal lacks the tenant/scope the target
    resource requires (production-hardening plan §5.1 / §7)."""


class PrincipalAccessDeniedError(SecurityError):
    """A sensitive operation was denied because no PrincipalContext was
    supplied, or the supplied Principal's tenant does not own the target
    resource. Surfaces the §5.4 fail-closed default: when tenant / scope /
    version cannot be confirmed, the operation is rejected rather than
    allowed on the strength of a guessable id alone (§5.5)."""


class ResourceError(LinktoolsAIError):
    """Base class for AssetStore-related errors."""


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


class InvalidAssetPathError(ResourceError):
    pass


class SkillResourceAccessError(ResourceError):
    """A skill-private resource path is forbidden: it is absolute, escapes the
    skill's ``agents/`` directory (including via symlink), is not Markdown, or
    is missing (spec ). Path safety is enforced on the resolved path, so a
    symlink that points outside ``agents/`` is rejected after resolve()."""


class SubagentResolutionError(LinktoolsAIError):
    """A ``call_subagent`` request could not be resolved: the named subagent is
    unknown, an ``instruction_path`` was given without an active skill, the
    active skill no longer exists / changed revision, or the request was
    malformed (spec //)."""


class StorageError(LinktoolsAIError):
    """Base class for Storage-facade-related errors."""


class StorageCapabilityError(StorageError):
    """Raised when an operation requires a StorageFeatures capability the
    active Storage does not expose (e.g. database-scoped transactions on
    FilesystemStorage, which is process-local)."""


class StorageRequirementsNotMetError(StorageCapabilityError):
    """Raised at build time by the RuntimeBuilder capability gate when the
    active Storage's StorageFeatures fall below a declared RuntimeRequirements
    minimum (e.g. process-local coordination configured for a topology that
    declared it needs distributed). Fail-fast, never a silent degradation."""


class StorageFeatureError(StorageCapabilityError):
    """Raised when a Storage's declared StorageFeatures do not match its wired
    objects -- a declared capability that has no backing object (e.g.
    streaming_blobs=True with no ArtifactStore, or a NONE transaction scope
    where a cross-store UoW was requested). Plan §4.5/§6.6 name this class as
    the unified signal for feature/behavior mismatch. The more specific
    :class:`StorageTransactionNotSupportedError` subclasses it so a caller
    catching ``StorageFeatureError`` sees both."""


class StorageTransactionNotSupportedError(StorageFeatureError):
    """features.transactions is TransactionScope.NONE or PROCESS_LOCAL on this
    Storage (no cross-store UoW available) but a caller requested an atomic
    cross-store write."""


class StorageConcurrencyNotSupportedError(StorageCapabilityError):
    """optimistic_concurrency is False but a caller requested CAS-style updates."""


class StorageLeaseNotSupportedError(StorageCapabilityError):
    """leasing is False but a caller (e.g. swarm claim) requested a lease."""


class IdempotencyConflictError(LinktoolsAIError):
    """Same idempotency key reused with a different request hash."""


class LostIdempotencyClaimError(LinktoolsAIError):
    """complete/fail did not match the persisted record (owner+generation no
    longer hold -- a newer worker stole the lease). The terminal write is
    rejected rather than silently succeeding."""


class IdempotencyConfigurationError(LinktoolsAIError):
    """An idempotent call lacks the context or trusted key needed for safety."""


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


class RunNotResumableError(RunError):
    """A run marked NON_RESUMABLE at creation time cannot be resumed (§13.7).
    Raised at resume entry instead of attempting a resume that could never be
    deterministic (unversioned handler / ephemeral provider / dynamic output /
    missing resource snapshot)."""


class ManifestDriftError(RunError):
    """The current environment no longer matches the ExecutionManifest the run
    was prepared against (§13.6) -- e.g. the resolved model provider's revision
    changed between prepare and resume. Raised by ManifestResolver.resolve;
    resume refuses rather than silently re-resolving against the drifted
    environment."""


class RunInvariantError(RunError):
    """A run completed without the authoritative state the runtime contract
    requires (e.g. no terminal RunResult after a non-pausing execute). Raised
    instead of fabricating an empty success result that would mask the bug."""


class RunPaused(RunError):
    """Raised by GovernedToolInvoker when a tool requires approval, and propagated
    through pydantic-ai's tool-execution stack out to AgentEngine, which
    persists the ApprovalRequest, checkpoints state, transitions the Run to
    WAITING_APPROVAL, and appends the pause events -- all atomically in one
    UnitOfWork on SqlAlchemy storage. This is the single approval path: the
    executor only emits the signal; it never persists approval state itself.

    This is a control-flow signal, NOT an error condition -- it's a RunError
    (not a ToolError) precisely so PolicyCapability.before_tool_execute (which
    only catches ToolDeniedError/ToolApprovalRequiredError -> SkipToolExecution)
    lets it propagate. AgentEngine catches it; nothing else should.

    ``approval_id`` is a fresh id minted here and then persisted by the
    GovernedToolInvoker no longer writes the ApprovalRequest itself; it only mints
    the id so the id it reports is the same one AgentEngine's suspension
    handler will actually persist. ``run_id`` is already resolved through
    GovernedToolInvoker.run_id_resolver. The remaining fields carry everything the
    suspension handler needs to construct and persist the ApprovalRequest
    without GovernedToolInvoker touching the ApprovalStore. Only primitive types are
    used here (no domain dataclass import) to keep this module dependency-free."""

    def __init__(
        self,
        run_id: str,
        approval_id: str,
        *,
        tool_call_id: "str | None" = None,
        tool_name: "str | None" = None,
        reason: "str | None" = None,
        arguments: "dict | None" = None,
        idempotency_key: "str | None" = None,
        binding: "dict | None" = None,
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
        self.binding = binding or {}


class SessionError(LinktoolsAIError):
    """Base class for Session-related errors."""


class SessionAccessDeniedError(SessionError):
    """A session exists but does not belong to the current principal/tenant.
    Raised by resolve_session when (user_id, tenant_id) do not match the
    session's owner -- the message never reveals whether the session belongs to
    someone else."""


class SessionSequenceConflictError(SessionError):
    """Raised when the SessionStore cannot reserve a unique message sequence
    after repeated conflicts (the store is the sole sequence
    authority, mirroring EventSequenceConflictError)."""


class SessionCorruptionError(SessionError):
    """A session record / message file is present but unreadable (truncated or
    malformed JSON). Distinct from "session does not exist": the file is
    preserved in place and the path is included so a repair tool can target it
    rather than the store silently masking corruption as a missing session."""


class EventError(LinktoolsAIError):
    """Base class for Event-related errors."""


class EventSequenceConflictError(EventError):
    pass


class ToolError(LinktoolsAIError):
    """Base class for Tool-execution-related errors."""


class ToolDeniedError(ToolError):
    pass


class ToolResultDeniedError(ToolDeniedError):
    """The tool ran, but its result was rejected by after-tool policy."""


class ToolApprovalRequiredError(ToolError):
    pass


class ToolPolicyResolutionError(ToolError):
    """A ToolPolicyProvider could not resolve a policy for a tool. The default
    posture is fail closed: the ManagedToolAdapter catches this, emits a
    SecurityDegraded event, and denies the call rather than running ungoverned."""


class ToolTimeoutError(ToolError):
    pass


class ToolSchemaError(ToolError):
    """Base for JSON-schema validation/definition errors. Downstream never sees
    a bare jsonschema.ValidationError / SchemaError / ImportError."""


class ToolSchemaDefinitionError(ToolSchemaError):
    """A tool's parameters_json_schema is itself malformed. Detected at assembly
    time (never postponed to first call). Never retried."""


class ToolSchemaValidationError(ToolSchemaError):
    """A tool's arguments (or result) failed JSON-schema validation -- e.g. a
    pipeline MODIFY produced arguments the tool cannot accept, or the original
    call's arguments did not match the declared parameters_json_schema. Never
    retried: the same payload will fail the same way."""


class PipelineExecutionError(ToolError):
    """A SecurityPipeline hook raised an unexpected exception. Pipelines fail
    closed; this is the stable error surfaced when a pipeline error cannot be
    attributed to a DENY decision. Never retried."""


class TransientToolError(ToolError):
    """A tool execution error that MAY succeed on retry (network blip, transient
    lock conflict, etc.). ManagedToolAdapter retries these up to max_retries."""


class ToolCommitError(ToolError):
    """The tool Handler ran (its side effect happened) but the fenced result
    commit could not be confirmed. The Handler MUST NOT be re-invoked.

    The idempotency record's resulting state depends on which step failed:
    if recording the execution receipt (``mark_executed``) could not be
    confirmed the record is UNKNOWN (outcome unknowable); if the receipt landed
    but the final ``complete`` failed, the record is left EXECUTED (recoverable
    -- a later claim replays it). Wraps the underlying failure
    (``__cause__``)."""


class ToolIdempotencyConflictError(ToolError):
    pass


class IdempotencyInProgressError(ToolError):
    """Raised by GovernedToolInvoker when an idempotent call hits a RESERVED record
    (another in-flight call owns the reservation). "wait / return
    in-progress / reject duplicate" are policy choices; for now the executor
    rejects -- the caller can retry once the in-flight call completes and the
    record moves to COMPLETED or FAILED."""


class PolicyError(LinktoolsAIError):
    """Base class for PolicyEngine-related errors."""


class ModelRoutingError(LinktoolsAIError):
    pass


class ModelInvocationDeniedError(LinktoolsAIError):
    """The model call was denied by before_model policy (DENY or an unsupported
    action). Raised before the delegate model is invoked, so no prompt leaves."""


class ModelResultDeniedError(LinktoolsAIError):
    """The model's result was denied or replaced by after_model policy. Raised
    before the un-audited result reaches the caller."""


class ModelPolicyExceededError(LinktoolsAIError):
    """Raised when a ModelPolicy limit (max_tokens, ...) is violated by a model
    call's actual usage. Carries ``kind`` so callers can distinguish which limit
    fired (currently only ``"max_tokens"``; ``"budget"`` requires a pricing
    cost-per-token rates exist)."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class SwarmError(LinktoolsAIError):
    """Base class for Swarm-related errors."""


class SwarmRunNotFoundError(SwarmError):
    pass


class SwarmResumeUnsupportedError(SwarmError):
    """The selected strategy has no explicit checkpoint-resume protocol."""

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


class MCPErrorCode(str, Enum):
    AUTHENTICATION = "authentication"
    CONNECTION = "connection"
    DISCOVERY_UNSUPPORTED = "discovery_unsupported"
    INVALID_TOOL_DEFINITION = "invalid_tool_definition"
    PROTOCOL = "protocol"


class MCPConnectionError(LinktoolsAIError):
    """An MCP server connection could not be established or was lost."""

    code = MCPErrorCode.CONNECTION


class MCPConnectionUnavailableError(MCPConnectionError):
    pass


class MCPAuthenticationError(MCPConnectionError):
    code = MCPErrorCode.AUTHENTICATION


class MCPDiscoveryError(MCPConnectionError):
    code = MCPErrorCode.PROTOCOL


class MCPDiscoveryUnsupportedError(MCPDiscoveryError):
    code = MCPErrorCode.DISCOVERY_UNSUPPORTED


class MCPToolDefinitionError(MCPDiscoveryError):
    code = MCPErrorCode.INVALID_TOOL_DEFINITION


class MCPToolError(LinktoolsAIError):
    """An MCP tool invocation failed at the protocol/transport layer."""

    code = MCPErrorCode.PROTOCOL


class ToolSecurityAuditError(ToolError):
    """A security-critical audit event could not be persisted."""


class ExtensionNotFoundError(CapabilityNotFoundError):
    pass


class ExtensionResourceNotFoundError(CapabilityNotFoundError):
    pass


class ExtensionResourceAccessDeniedError(PolicyError):
    """An extension resource path was outside the allowed scope/extension set."""


class ExtensionEntrypointNotFoundError(CapabilityNotFoundError):
    pass


class ExtensionEntrypointDeniedError(PolicyError):
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


class ToolBindingError(LinktoolsAIError):
    """A tool execution cannot be bound to stable revisions."""


class ApprovalNotFoundError(ApprovalError):
    pass


class ApprovalConflictError(ApprovalError):
    pass


class InvalidApprovalTransitionError(ApprovalError):
    pass
