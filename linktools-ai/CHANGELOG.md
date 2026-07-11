# Changelog

## Phase 2A — Security Execution Closed Loop

### Added

- **SecurityBaseline** (`linktools.ai.security`): domain-agnostic default
  safety baseline. Enabled by default; callers can override or disable via
  `Runtime.build(security=SecurityBaseline(enabled=False))`. Carries a
  `CommandPolicy` (high-risk terminal denylist) and an optional
  `SecurityPipeline`.

- **SecurityPipeline** Protocol: formal extension point for downstream safety
  audit/decision. Hooks: `before_tool` / `after_tool` / `on_security_event`
  (+ `before_model`/`after_model` reserved for Phase 2A.2). Returns
  `PipelineDecision` (ALLOW / DENY / REQUIRE_APPROVAL / MODIFY / AUDIT_ONLY).
  `CompositeSecurityPipeline` composes multiple pipelines with strict
  precedence (DENY > APPROVAL > MODIFY > ALLOW).

- **ManagedToolAdapter**: the single entry point through which every
  model-driven tool call passes. Chain: descriptor lookup -> ToolPolicyProvider
  resolve -> SecurityBaseline merge -> SecurityPipeline.before_tool -> handler
  execution with timeout + retry -> SecurityPipeline.after_tool -> stable
  error/result.

- **ManagedToolsetWrapper**: wraps opaque toolsets (e.g. pydantic-ai
  `MCPToolset`) so every `call_tool` goes through the pipeline governance chain.

- **ToolDescriptor**: structured metadata classifying a tool (name, source,
  category, risk, mutating, capability_kind/name). Avoids guessing risk from
  function names.

- **ToolContribution**: pairs a toolset with its explicit descriptors so the
  assembler + adapter never need toolset introspection.

- **ResolvedToolPolicy + ToolPolicyProvider**: the merged policy for a single
  tool invocation. `MetadataBackedPolicyProvider` bridges existing
  `ToolRegistry` metadata to the new Protocol.

- **TransientToolError**: errors explicitly marked as retryable.

### Changed

- **Default Agent output** is now plain text (`str`) when no `output_schema` is
  declared (previously defaulted to `dict`). Existing tests pass unchanged.

- **AgentRunner** routes all capability-assembled tools through
  `ManagedToolAdapter` when a `SecurityBaseline` is enabled (the default).
  `SecurityBaseline(enabled=False)` falls back to the legacy direct-toolset
  path.

- **MCPDiscoveryMode**: `MCPServerSpec.discovery_mode` defaults to `"strict"`.
  MCPProvider fails closed when strict mode + governance config
  (enabled/disabled/prefix) is present but live tool enumeration returns empty.

- **Subagent identity propagation**: child runs now inherit `user_id` /
  `tenant_id` / `workspace` from the parent RunContext.

- **Concurrency validation**: `max_concurrency >= 1`, `max_depth >= 1`,
  `timeout_seconds > 0` enforced at resolution time.

### Migration

To disable the default security baseline:

```python
runtime = Runtime.build(
    storage=storage,
    security=SecurityBaseline(enabled=False),
)
```

To inject a custom pipeline:

```python
runtime = Runtime.build(
    storage=storage,
    security=SecurityBaseline(
        pipeline=CompositeSecurityPipeline([my_audit_pipeline, my_dlp_pipeline]),
    ),
)
```
# Unreleased

- Hardened JSON Schema validation, MCP exposed-name handling, registry policy
  mapping, idempotency, audit failure behavior, and recursive audit redaction.
- MCP raw tool aliases are no longer registered as model-visible tools.
- Invalid schemas, missing idempotency context, and security-audit failures now
  fail closed by default.
- Oversized security events are persisted as a `TruncatedSecurityEvent`
  dataclass instead of a plain dict, so `FileEventStore` no longer raises
  `TypeError` and the audit trail survives in both `fail_closed` and
  `best_effort` modes.
- `idempotency_strategy: business_key` now requires `idempotency_key_field` at
  declaration time (registry load, `ResolvedToolPolicy`, and
  `EffectiveToolPolicy`); the misconfiguration is rejected before any tool call
  rather than surfacing inside the idempotency key builder.
- `Runtime.inspect()` injects a collecting security-event emitter, so an MCP
  provider resolving under best-effort discovery degrades the same way under
  inspection as under a real run (no hard failure) and records the degradation
  as an inspection warning.
- `Runtime` no longer exposes a public `capability_assembler` property;
  `Runtime.inspect()` is the single public entry and returns only safe
  `ToolDescriptor` snapshots (no raw handlers).
- MCP tool `description` and `readOnlyHint` are preserved end to end:
  `ManagedToolDefinition` carries a `description` field that reaches the model
  tool definition, and only an explicit `readOnlyHint=true` marks a descriptor
  non-mutating (unknown stays high-risk/mutating).
- Security vs observability audit events are now classified explicitly at each
  emission site instead of by payload class name, so a new audit event can no
  longer be silently misrouted to the observability channel.
