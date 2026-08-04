#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""errors.py: stable domain error hierarchy. Never identify an error by string
matching -- always by type."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tasks.models import TaskUsage


class LinktoolsAIError(Exception):
    """Base class for every error raised by linktools.ai."""


class JsonEncodingError(TypeError):
    """Raised when a value cannot be normalized to canonical JSON.

    Subclasses TypeError so callers that catch the generic JSON-conversion
    TypeError (and ``pytest.raises(TypeError)``) keep working, while the name
    makes the failure explicit at the persistence boundary."""


class RuntimeInitializationError(LinktoolsAIError):
    """The runtime cannot safely initialize a required component."""


class UnsafeSandboxError(RuntimeInitializationError):
    """A trusted-local backend was used where tenant isolation is required."""


class SecurityError(LinktoolsAIError):
    """Security-domain failures: a sensitive operation was attempted without a
    valid Principal, or the Principal lacks the tenant/scope the target
    asset requires."""


class PrincipalAccessDeniedError(SecurityError):
    """A sensitive operation was denied because no PrincipalContext was
    supplied, or the supplied Principal's tenant does not own the target
    asset. Surfaces the fail-closed default: when tenant / scope /
    version cannot be confirmed, the operation is rejected rather than
    allowed on the strength of a guessable id alone."""


class AssetError(LinktoolsAIError):
    """Base class for skill-asset-related errors."""


class SkillAssetAccessError(AssetError):
    """A skill-private asset path is forbidden: it is absolute, escapes the
    skill's ``agents/`` directory (including via symlink), is not Markdown, or
    is missing. Path safety is enforced on the resolved path, so a
    symlink that points outside ``agents/`` is rejected after resolve()."""


class SubagentResolutionError(LinktoolsAIError):
    """A ``call_subagent`` request could not be resolved: the named subagent is
    unknown, an ``instruction_path`` was given without an active skill, the
    active skill no longer exists / changed revision, or the request was
    malformed (spec //)."""


class ArtifactError(LinktoolsAIError):
    """Base class for Artifact-domain errors (content-addressed blobs and the
    immutable records that pin them)."""


class ArtifactRecordConflictError(ArtifactError):
    """An :class:`ArtifactRecord` id already exists with different content.
    Records are create-only: the same id with byte-identical content is
    idempotent, but a different sha256 / tenant / provenance is refused rather
    than overwriting the lineage of the prior write."""


class ArtifactRecordCorruptError(ArtifactError):
    """A stored ArtifactRecord cannot be trusted: it is unparseable, missing
    required fields, has a sha256 that is not a valid digest, or its id/tenant
    does not match the path it was filed under. Raised fail-closed so the orphan
    sweeper never mistakes a broken record for an unreferenced blob."""


class InvalidArtifactDigestError(ArtifactError):
    """A digest did not parse as exactly 64 lowercase hex characters. Raised at
    the domain boundary before the digest becomes a coordinator key, a lock-file
    path component, or a blob address, so traversal / overflow / collision input
    can never reach those surfaces. The message intentionally does not echo the
    rejected value."""


class StorageError(LinktoolsAIError):
    """Base class for Storage-facade-related errors."""


class StorageConflictError(StorageError, ValueError):
    """A compare-and-swap, lease, or revision update lost a race."""


class RecoveryConflictError(StorageConflictError):
    """Another owner won the exclusive recovery claim."""


class StorageCorruptionError(StorageError):
    """Required local or database persistence data is missing or malformed."""


class InvalidStoragePathError(StorageError):
    """A caller supplied an identifier that cannot safely address local data."""


class StorageFeatureSupportError(StorageError):
    """Raised when an operation requires a StorageFeatures flag the
    active Storage does not expose (e.g. database-scoped transactions on
    local directory storage, which is process-local)."""


class StorageRequirementsNotMetError(StorageFeatureSupportError):
    """Raised at build time by the RuntimeBuilder storage-feature gate when the
    active Storage's StorageFeatures fall below a declared RuntimeRequirements
    minimum (e.g. process-local coordination configured for a topology that
    declared it needs distributed). Fail-fast, never a silent degradation."""


class StorageFeatureDeclarationError(StorageFeatureSupportError):
    """A wired component does not expose the public storage-feature properties the
    Storage's feature derivation requires (e.g. an ArtifactStore stand-in
    missing ``record_store`` / ``supports_streaming`` / ``coordination_scope``).
    Fail-closed at construction: a Storage cannot declare a feature none of
    its wired objects actually provides."""


class StorageFeatureError(StorageFeatureSupportError):
    """Raised when a Storage's declared StorageFeatures do not match its wired
    objects -- a declared feature that has no backing object (e.g.
    streaming_blobs=True with no ArtifactStore, or a NONE transaction scope
    where a cross-store UoW was requested). This class is the unified signal
    for feature/behavior mismatch. The more specific
    :class:`StorageTransactionNotSupportedError` subclasses it so a caller
    catching ``StorageFeatureError`` sees both."""


class StorageTransactionNotSupportedError(StorageFeatureError):
    """features.transaction_scope is TransactionScope.NONE or PROCESS_LOCAL on this
    Storage (no cross-store UoW available) but a caller requested an atomic
    cross-store write."""


class StorageConcurrencyNotSupportedError(StorageFeatureSupportError):
    """optimistic_concurrency is False but a caller requested CAS-style updates."""


class StorageLeaseNotSupportedError(StorageFeatureSupportError):
    """leasing is False but a caller (e.g. swarm claim) requested a lease."""


class StorageCoordinationNotSupportedError(StorageFeatureSupportError):
    """Raised when a KeyedCoordinator that cannot provide the required
    coordination scope is constructed (e.g. a filesystem flock coordinator on
    a non-POSIX platform), or when a deployment that needs a distributed
    coordinator did not inject one. Fail-closed: never silently degrade to a
    lockless fallback."""


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


class RunIdentityConflictError(RunConflictError):
    """A run id was reused with a different complete start identity."""


class ChildRunAlreadyActiveError(RunConflictError):
    """A persisted child is already being driven by another execution."""


class RunCancelledError(RunError):
    pass


class ExecutionAlreadyActiveError(RunConflictError):
    """A second live task attempted to drive the same execution."""


class InvalidRunTransitionError(RunError):
    pass


class RunNotResumableError(RunError):
    """A run marked NON_RESUMABLE at creation time cannot be resumed.
    Raised at resume entry instead of attempting a resume that could never be
    deterministic (unversioned handler / ephemeral provider / dynamic output /
    missing asset snapshot)."""


class ManifestDriftError(RunError):
    """The current environment no longer matches the ExecutionManifest the run
    was prepared against -- e.g. the resolved model provider's revision
    changed between prepare and resume. Raised by ManifestResolver.resolve;
    resume refuses rather than silently re-resolving against the drifted
    environment."""


class RunInvariantError(RunError):
    """A run completed without the authoritative state the runtime contract
    requires (e.g. no terminal RunResult after a non-pausing execute). Raised
    instead of fabricating an empty success result that would mask the bug."""


class RunStateError(RunError):
    """A persisted run state cannot be driven by the requested operation."""


class RunDefinitionError(RunError):
    """A persisted runnable definition has an unsupported schema."""


class RunDefinitionIntegrityError(RunDefinitionError):
    """A persisted runnable definition no longer matches its content hash."""


class RunLiveStreamAlreadyOpenError(RunError):
    """Raised by RunLiveEventHub.open when a live stream handle for the same
    run_id is already active. A run has at most one live consumer at a time;
    a second concurrent open would silently split the event fan-out between
    two handles, so this is refused rather than allowed to race."""


class RunLiveStreamClosedError(RunError):
    """Raised when a publish loses the race against close (the close-event
    fired mid-publish), or when a publish is attempted on an already-closed
    handle. Closes are signaled by an asyncio.Event (not a sentinel pushed
    into the queue), so a publish racing close MUST detect the loss and
    surface it -- a silent drop would let the caller believe the event was
    delivered to a consumer that has already given up."""


class RunPaused(RunError):
    """Raised by GovernedToolInvoker when a tool requires approval, and propagated
    through pydantic-ai's tool-execution stack out to AgentEngine, which
    persists the ApprovalRequest, checkpoints state, transitions the Run to
    PAUSED, and appends the pause events -- all atomically in one
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
    without GovernedToolInvoker touching the approval store. Only primitive types are
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
    """Raised when the session store cannot reserve a unique message sequence
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
    """A ToolPolicyResolver could not resolve a policy for a tool. The default
    posture is fail closed: ToolExecutionService catches this, emits a
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
    lock conflict, etc.). ToolExecutionService retries these up to max_retries."""


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


class ModelRetryConfigurationError(LinktoolsAIError):
    """A prebuilt (already-constructed) Model was resolved under a policy that
    asks the framework to configure provider retries (``request_retries`` is an
    int). A prebuilt model owns its own HTTP client, so the framework cannot set
    ``max_retries`` on it; the combination is rejected rather than silently
    ignored. ``request_retries=None`` is the signal that a prebuilt model manages
    its own retry behavior."""


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


class SwarmCommitCoordinatorUnavailableError(SwarmError):
    """build_runtime_components could not dispatch a SwarmCommitCoordinator
    from the Storage. SwarmEngine takes the coordinator as a required dep,
    so a Storage that exposes neither DATABASE transaction scope nor a
    filesystem root fails the build rather than silently running without
    commit-log idempotency."""


class SwarmResumeUnsupportedError(SwarmError):
    """The selected strategy has no explicit checkpoint-resume protocol."""

    pass


class SwarmStepNotFoundError(SwarmError):
    pass


class SwarmStepConflictError(SwarmError):
    pass


class SwarmConflictError(SwarmError):
    pass


class InvalidSwarmTransitionError(SwarmError):
    pass


class SwarmLimitExceededError(SwarmError):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ChildExecutionPlatformError(SwarmError):
    """A child execution raised outside the expected Agent outcome path."""

    child_run_id: str
    usage: "TaskUsage"
    error_type: str
    safe_message: str
    cause: BaseException = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.safe_message)

    def add_note(self, note: str) -> None:
        """Keep BaseException note semantics compatible with frozen dataclasses."""
        if not isinstance(note, str):
            raise TypeError("note must be a str")
        notes = getattr(self, "__notes__", ())
        object.__setattr__(self, "__notes__", [*notes, note])


class ChildRunMissingError(SwarmError):
    """The deterministic child record was not present after execution failed."""


class ChildSnapshotError(SwarmError):
    """The child record had no decodable persisted snapshot."""


class ChildCancelNotConvergedError(SwarmError):
    """A recovery cancellation did not reach a child terminal state in time."""


class NodeLeaseLostError(SwarmError):
    """The scheduler lost ownership of a claimed task execution."""


class ParentLeaseLostError(SwarmError):
    """The scheduler lost ownership of the parent run lease."""


class ParentLeaseGuardError(SwarmError):
    """A child start was rejected by the parent's owner/fence/lease guard."""


class ParentTerminalGateError(SwarmError):
    """A parent cannot enter a terminal state while its graph is not converged."""


class SwarmConvergenceError(SwarmError):
    """A swarm could not safely converge its children or cleanup state."""

    def __init__(
        self,
        message: str,
        *,
        primary_error: "BaseException | None" = None,
        cleanup_error: "BaseException | None" = None,
        diagnostics: "tuple[CleanupDiagnostic, ...]" = (),
    ) -> None:
        super().__init__(message)
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        self.diagnostics = diagnostics


class UsageRegressionError(SwarmError):
    """An authoritative cumulative usage snapshot moved backwards."""


class UsageObservationConflictError(SwarmError):
    """The same usage revision or request key carried different data."""


class TaskGraphInvariantError(SwarmError):
    """The DAG cannot make progress: no node is ready, in flight, or skippable,
    yet not all nodes are terminal. A run-level failure, not a node failure."""


@dataclass(frozen=True, slots=True)
class CleanupDiagnostic:
    stage: str
    node_id: "str | None"
    error_type: str
    safe_message: str


@dataclass(frozen=True, slots=True)
class TaskGraphCleanupError(SwarmError):
    """Primary graph failure plus cleanup failure that requires recovery."""

    primary_error: "BaseException | None"
    cleanup_error: BaseException
    diagnostics: "tuple[CleanupDiagnostic, ...]"

    def __post_init__(self) -> None:
        Exception.__init__(self, "task graph cleanup failed")


def _set_cleanup_error_attribute(self, name: str, value: object) -> None:
    if name in {"primary_error", "cleanup_error", "diagnostics"} and hasattr(
        self, name
    ):
        raise AttributeError(f"{name} is immutable")
    object.__setattr__(self, name, value)


TaskGraphCleanupError.__setattr__ = _set_cleanup_error_attribute


class MemoryError(LinktoolsAIError):
    """Base class for Memory-related errors."""


class MemoryNotFoundError(MemoryError):
    pass


class MemoryConflictError(MemoryError):
    pass


class SpecError(LinktoolsAIError):
    """Base class for specification loading and parsing errors."""


class SpecNotFoundError(SpecError):
    pass


class SpecConflictError(SpecError):
    pass


class SpecParseError(SpecError):
    pass


class InvalidSpecError(SpecError):
    """A parsed spec is structurally present but semantically invalid."""


# --- Feature resolution tree --------------------------------------------
# Resolving AgentSpec.features into concrete contributions can fail in two
# qualitatively different ways: a referenced feature cannot be found, or two
# features collide. Both carry agent_id / ref so callers can pinpoint the
# failing declaration instead of grepping strings.


class AgentAssemblyError(LinktoolsAIError):
    """Base class for feature-resolution failures (assemble-time)."""


class AgentFeatureNotFoundError(AgentAssemblyError):
    pass


class AgentFeatureConflictError(AgentAssemblyError):
    """Two features conflict at the declaration or contribution boundary."""


class ToolConflictError(LinktoolsAIError):
    """Tool exposure produced duplicate names or exceeded a configured limit."""


class SkillNotFoundError(AgentFeatureNotFoundError):
    pass


class MCPServerNotFoundError(AgentFeatureNotFoundError):
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


class ExtensionNotFoundError(AgentFeatureNotFoundError):
    pass


class ExtensionContentNotFoundError(AgentFeatureNotFoundError):
    pass


class ExtensionContentAccessDeniedError(PolicyError):
    """An extension asset path was outside the allowed scope/extension set."""


class ExtensionEntrypointNotFoundError(AgentFeatureNotFoundError):
    pass


class ExtensionEntrypointDeniedError(PolicyError):
    """An entrypoint kind/name was not on the declared allowlist."""


class SubagentNotFoundError(AgentFeatureNotFoundError):
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
