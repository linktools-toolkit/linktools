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
