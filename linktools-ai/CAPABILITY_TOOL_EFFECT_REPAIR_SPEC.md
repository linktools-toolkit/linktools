# Capability Tool-Effect Semantics Repair Spec

## Status

- Branch: `fix/ai-capability-tool-effect-semantics`
- Scope: `linktools-ai`
- Repair type: runtime correctness, durable-recovery contract repair, and Pydantic-native user-content restoration
- No database schema migration
- Public input extension: `AgentHandle.start/run/stream/task` accept Pydantic AI `UserContent` sequences in addition to plain strings
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
16. Agent callers may supply the same `str | Sequence[UserContent]` shape accepted by Pydantic AI. Linktools must not define parallel attachment classes.
17. The Agent boundary converts rich user content once to canonical Pydantic model-message JSON and tags it with the explicit internal codec `pydantic-user-content-v1`. Plain strings use codec `text`. Ordinary text is never interpreted by content prefix or other magic bytes.
18. Runtime request contracts remain string-based. Every durable boundary that can erase the in-memory string subtype must persist the codec explicitly: TaskGraph stores `user_prompt_codec`, Temporal request objects store `user_prompt_codec`, and Recovery uses its existing `StoredPayload` discriminator (`utf-8` for text; JSON `{codec,value}` for rich content). Historical records without the new optional discriminator read as `text`.
19. Runtime-generated TaskGraph suffix text must preserve the original codec and restore as an additional textual `UserContent` item. Planner code must not understand attachment types.
20. `AgentExecutor` is the sole native restoration point: immediately before `run_stream_events()`, rich transport is structurally validated and restored to native Pydantic AI `UserContent`; plain strings remain plain strings.
21. Prompt transport integrity is owned by the existing durable container (`StoredPayload` / ObjectStore content digest) plus structural codec validation. Do not add a second digest inside prompt text and do not require a newer Pydantic version to re-encode historical content byte-for-byte identically after decoding it.
22. Provider-owned `UploadedFile` references are not accepted as durable Agent input until provider lifetime, portability and recovery semantics are defined. Inline `BinaryContent` and other self-contained or durable URL content remain governed by the Pydantic message codec and existing Runtime size limits.
23. Durable prompt compatibility requires frozen historical fixtures. Same-version encode/decode tests are insufficient; legacy records without codec fields and historical rich payloads must remain readable according to their frozen codec contract.
24. Any discriminator that changes semantic identity must participate in stable hashes and idempotency identities. Rich prompts therefore hash `{codec,value}` while historical `text` prompts retain the original raw-string digest shape.

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
- Skill missing-ID regression passes without expanding the capability package public API;
- cancellation tests pass;
- repository search finds no second built-in replay-safety truth source;
- Pydantic `BinaryContent` user input produces deterministic `pydantic-user-content-v1` transport and restores natively;
- ordinary text, including text resembling historical transport prefixes, remains codec `text` and is never reinterpreted;
- malformed rich payloads fail closed as `STORAGE_INTEGRITY_ERROR`, while unknown codecs fail as `STORAGE_VERSION_UNSUPPORTED`;
- provider-owned `UploadedFile` input is rejected with an explicit durable-lifecycle reason;
- TaskGraph-style runtime suffix text preserves the rich codec and restores as an additional text content part;
- TaskGraph admission, Temporal request persistence and Recovery checkpoints preserve explicit codec information, while frozen legacy records without codec fields remain readable as text;
- prompt compatibility includes frozen historical fixtures rather than only current encode/decode round trips;
- text prompt request digests remain backward-stable, while identical payload bytes with different semantic codecs produce distinct request digests;
- Runtime public request schemas remain string-based and no Linktools attachment DTO hierarchy is introduced;
- `python manage.py check linktools-ai` passes on the repository CI Python matrix.
