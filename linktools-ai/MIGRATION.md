# Migration notes

- MCP allow/deny lists continue to use server `raw_name`; model-visible names
  and policy lookups use `exposed_name`. Raw aliases are not registered.
- `CapabilityRuntimeOptions.tool_exposure` is merged independently from other
  runtime options. An explicit value overrides the baseline; omitted values do
  not disable a baseline exposure policy.
- Tool registry entries may set `enabled`, `max_retries`, `schema_version`, and
  the explicit `idempotency_strategy`/`idempotency_key_field` pair.
- Schema versions use provider > definition > `"1"` precedence; they are not
  selected by string sorting.
- `business_key` idempotency requires a trusted configured field. Missing keys,
  run context, or persistent storage are errors rather than a non-idempotent
  fallback.
- Custom MCP managers must implement the explicit discovery and
  `call_tool(connection_ref=...)` protocol. Older managers can be wrapped with
  `LegacyMCPConnectionManagerAdapter(..., empty_is_verified=...)`.
- MCP raw toolsets are no longer returned from capability bundles. After-tool
  audit events now precede the final `ToolCompleted` event and record whether a
  result was returned, modified, or denied.
- Migrate public capability inspection from `Runtime.assemble()` to
  `Runtime.inspect()`. Direct `AgentRunner` callers must inject the assembler
  and managed executor, or use `Runtime.build()`.
- `Runtime` no longer exposes a public `capability_assembler` property. The
  assembler carries raw executable handlers and is internal; reach resolved
  capabilities through `Runtime.inspect()`, which returns only safe
  `ToolDescriptor` snapshots.
- `idempotency_strategy: business_key` must declare `idempotency_key_field`.
  The combination is rejected at registry load and at `ResolvedToolPolicy` /
  `EffectiveToolPolicy` construction, not deferred to the first tool call.
- An MCP tool is marked non-mutating only when its server annotations set
  `readOnlyHint=true`; a read-only-looking name does not imply non-mutating,
  and an unknown hint stays high-risk/mutating.
- Oversized security audit events are persisted as a `TruncatedSecurityEvent`
  dataclass (original payload dropped, size recorded); they never break the
  event store contract.
