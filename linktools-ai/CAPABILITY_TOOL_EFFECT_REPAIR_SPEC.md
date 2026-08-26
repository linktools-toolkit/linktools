# Capability Tool-Effect Semantics Repair Spec

## Status

- Branch: `fix/ai-capability-tool-effect-semantics`
- Scope: `linktools-ai`
- Repair type: runtime correctness and durable-recovery contract repair
- No database schema migration
- No new public API
- No pessimistic locking

## Required invariants

1. Linktools is the single owner of built-in tool provenance and durable effect policy. Runtime persistence must not independently infer built-in replay safety from optional tool metadata.
2. Model-visible failures (`ValidationError`, `ModelRetry`, `ToolRetryError`, `ToolFailed`, `ToolFailedError`) describe what the model should observe; they do **not** by themselves prove that the handler produced no side effect. They become durable `FAILED` attempts only when the handler did not enter, the tool is effect-free, or the operation is explicitly replay-safe. An entered replay-unsafe effectful handler fails closed as `EFFECT_UNKNOWN` / `TOOL_EFFECT_UNKNOWN`.
3. A durable failure commit that is itself uncertain remains a storage/recovery error. It must not be reclassified as tool-effect uncertainty.
4. `SkipToolExecution` is an intentional successful short-circuit: persist the supplied result as `COMPLETED` and complete the Step tool-effect record exactly once.
5. Linktools does not currently implement Pydantic AI dynamic-deferred resume/result handoff. A resolvable `CallDeferred` / `ApprovalRequired` raised after entering the execution hook is terminalized as `FAILED` with `CAPABILITY_POLICY_CONFLICT` and `reason=dynamic_deferred_unsupported`; if a replay-unsafe effectful handler has already entered, fail closed as `TOOL_EFFECT_UNKNOWN` instead. Static/declarative deferral that Pydantic handles before execution hooks is unaffected.
6. Generic handler errors use the effective tool policy:
   - effect-free -> `FAILED`;
   - replay-safe and effectful -> preserve `CLAIMED`, raise `STORAGE_RECOVERY_REQUIRED`;
   - replay-unsafe and effectful -> `EFFECT_UNKNOWN`, raise `TOOL_EFFECT_UNKNOWN`.
7. Cancellation must not weaken effect truth: an entered replay-unsafe operation remains unknown; replay-safe operations remain recoverable.
8. Tool-operation result/failure cache replay must preserve Pydantic AI control-flow semantics. New `ToolRetryError` and `ToolFailedError` payloads preserve their complete message parts; historical payload formats remain readable.
9. Trusted built-in names are provenance-checked. A custom capability cannot acquire built-in safety by reusing a trusted tool name.
10. MCP and custom RuntimeCapability tools remain unsafe by default; existing `linktools.ai.replay_safe=True` remains the explicit strong opt-in for replay safety, but does not imply that the tool is effect-free.
11. Linktools plan mode exposes only the Linktools-owned planning surface. Harness dependency expansion must not silently add new planning tools; currently the surface is `write_plan` only.
12. Unknown Skill IDs are model-correctable and raise `ModelRetry`, not a generic infrastructure exception.
13. Normal handled tool failures must not emit duplicate detached-handler cleanup errors.
14. Existing ToolOperation, Session, Execution, Step and database/file schemas remain unchanged.
15. This correctness repair does not bump the runtime contract revision.

## Built-in policy matrix

| Capability/tool | replay-safe | effect-free |
|---|---:|---:|
| Filesystem `file_info/find_files/list_directory/read_file/search_files` | yes | yes |
| Filesystem `create_directory/edit_file/write_file` | no | no |
| Shell `check_command` | yes | yes |
| Shell `run_command/start_command/stop_command` | no | no |
| Memory `read_memory/search_memory` | yes | yes |
| Memory `write_memory/delete_memory` | yes | no |
| Skill `list_skills/load_skill` | yes | yes |
| Planning `write_plan` | yes | no |
| Subagent `delegate_task` | yes | no |
| MCP/custom default | no | no |
| MCP/custom explicit replay-safe metadata | yes | no |

`effect-free` means that an ordinary handler failure cannot leave an externally relevant mutation that Linktools must reconcile. `replay-safe` means Linktools may safely repeat the operation after an uncertain attempt. These are separate properties and must not be inferred from each other.

## Validation gates

The repair is complete only when all of the following hold:

- missing Harness `read_file` -> failed tool effect + model retry, never `TOOL_EFFECT_UNKNOWN`;
- model-visible failure matrix respects effect/replay policy instead of exception type alone;
- replay-unsafe effectful handlers remain fail-closed even when they raise `ModelRetry` / `ToolFailed`;
- generic effect-free / replay-safe effectful / replay-unsafe effectful state transitions pass;
- failure-commit uncertainty is not reclassified;
- `SkipToolExecution` success is terminalized exactly once;
- dynamic deferred-control paths are explicitly rejected without leaving unresolved claimed effects, while unsafe entered effects remain unknown;
- `ToolRetryError` and `ToolFailedError` durable payload round-trip tests pass;
- Planning exposes no accidental Harness granular tools;
- Skill missing-ID regression passes without expanding the public capability API;
- cancellation tests pass;
- repository search finds no second built-in replay-safety truth source;
- `python manage.py check linktools-ai` passes on the repository CI Python matrix.
