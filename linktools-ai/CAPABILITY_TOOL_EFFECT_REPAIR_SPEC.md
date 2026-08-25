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
2. Model-visible known failures (`ValidationError`, `ModelRetry`, `ToolRetryError`, `ToolFailed`, `ToolFailedError`) are durable `FAILED` tool attempts independently of replay safety. They must never become `TOOL_EFFECT_UNKNOWN` solely because the tool is effectful.
3. A durable failure commit that is itself uncertain remains a storage/recovery error. It must not be reclassified as tool-effect uncertainty.
4. `SkipToolExecution` is an intentional successful short-circuit: persist the supplied result as `COMPLETED` and complete the Step tool-effect record exactly once.
5. Dynamic `CallDeferred` / `ApprovalRequired` keeps a non-terminal claimed operation only when the handler did not enter or exact replay is safe. If an unsafe effectful handler already entered, fail closed as `TOOL_EFFECT_UNKNOWN`.
6. Generic handler errors use the effective tool policy:
   - effect-free -> `FAILED`;
   - replay-safe and effectful -> preserve `CLAIMED`, raise `STORAGE_RECOVERY_REQUIRED`;
   - replay-unsafe and effectful -> `EFFECT_UNKNOWN`, raise `TOOL_EFFECT_UNKNOWN`.
7. Cancellation must not weaken effect truth: an entered replay-unsafe operation remains unknown; replay-safe operations remain recoverable.
8. Tool-operation result/failure cache replay must preserve Pydantic AI control-flow semantics. New `ToolRetryError` and `ToolFailedError` payloads preserve their complete message parts; historical payload formats remain readable.
9. Trusted built-in names are provenance-checked. A custom capability cannot acquire built-in safety by reusing a trusted tool name.
10. MCP and custom RuntimeCapability tools remain unsafe by default; existing `linktools.ai.replay_safe=True` remains the explicit strong opt-in for custom tools.
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
| Planning `write_plan` | yes | yes |
| Subagent `delegate_task` | yes | no |
| MCP/custom default | no | no |
| MCP/custom explicit replay-safe metadata | yes | yes |

`effect-free` here means that an ordinary handler exception cannot leave an externally relevant mutation that Linktools must reconcile. It does not mean the tool has no in-process bookkeeping.

## Persistence evolution constraint

Minor representation changes must not make an existing Runtime store globally unreadable. Tolerant internal-record evolution and exact durable contracts remain separate: identity, authorization, digest, idempotency, state-machine and exact execution/recovery binding changes still fail closed and require explicit versioning or migration when semantics change.

## Validation gates

The repair is complete only when all of the following hold:

- missing Harness `read_file` -> failed tool effect + model retry, never `TOOL_EFFECT_UNKNOWN`;
- known failure matrix passes for replay-safe and replay-unsafe tools;
- generic effect-free / replay-safe effectful / replay-unsafe effectful state transitions pass;
- failure-commit uncertainty is not reclassified;
- SkipToolExecution success is terminalized exactly once;
- deferred-control paths satisfy the safe replay rule;
- ToolRetryError and ToolFailedError durable payload round-trip tests pass;
- Planning exposes no accidental Harness granular tools;
- Skill missing-ID regression passes;
- cancellation tests pass;
- repository search finds no second built-in replay-safety truth source;
- `python manage.py check linktools-ai` passes on the repository CI Python matrix.
