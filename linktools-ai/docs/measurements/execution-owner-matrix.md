# Execution field ownership

## Delivery status

This is the implementation coverage map, not a validation report. It describes
accepted first-version code and explicitly identified integration work. No CI,
performance, restore, or corruption-test pass is asserted by this document.
The in-memory `ExecutionRecord` remains a hydrated API contract; its fields do
not imply that those values are duplicated in the physical execution row.

Implementation status: base `3093c958` and completion `d3e56ff8` are
integrated (all written, not validated):

- Canonical `agent_run` admissions and `tool_operation` rows have Step-owned
  physical owners; generic State reads retain their logical interface.
- Tool arguments, result and error payloads reference the same immutable Step
  payload owner used by execution payloads. This supersedes the intermediate
  Step-ledger-inline representation; there is no body-size cutoff.
- Execution input/output values of every type and size have fixed Step payload
  references. Eligible exact Native input/output locations omit duplicate
  digest/size fields.
- Binding references no longer carry a second binding summary.
- All eight Runtime policy fields use a fixed accepted policy owner plus actual
  changed-field overrides. The policy owner is Runtime content, not Step content.
- Complete pending-continuation, terminal-handoff and result templates, and
  terminal-details references, have been written and integrated.

Completion commit `d3e56ff8` is also integrated:

- Arguments nested inside the pending template use fixed references to the same
  shared Step payload owner, rather than remaining duplicate inline arguments.
- Terminal-handoff error data shares the terminal-details owner instead of
  retaining an exact second error copy.

These two boundaries are implemented, not still implementation tasks. Their
combined publication and runtime validation have not been established here.

This is a code-status snapshot, not evidence that publication, CI, or the combined
runtime paths have passed. The publication commit and CI result are separate.

## Complete ExecutionRecord field map

`Control` means execution-specific identity, authorization, mutable coordination,
or an execution-level publication fact. It is not a cached copy of Step state.
`Fixed reference` means the execution retains a durable location, not another
summary of facts owned there. `Policy` is the fixed Runtime policy owner plus
only actual per-execution deviations. All eight policy fields use that same
owner, not eight owners.

| Field | Intended persisted owner / representation | Reason and implementation status |
| --- | --- | --- |
| `execution_id` | Control | Runtime operation identity; one execution may have multiple Step runs. Retained. |
| `session_id` | Fixed Session identity | Associates the execution with its session, not a copy of session state. Retained. |
| `parent_execution_id` | Fixed Execution identity | Execution lineage is not Step run lineage. Retained. |
| `root_execution_id` | Fixed Execution identity | Stable execution-tree routing identity. Retained. |
| `previous_execution_id` | Fixed Execution identity | Continuation relationship, not the previous Step event. Retained. |
| `fork_base_execution_id` | Fixed Execution identity | Identifies the execution fork source. Retained. |
| `lineage_kind` | Control | Distinguishes execution run/continuation/fork/subagent admission. Retained. |
| `status` | Control | Authoritative Runtime state machine, including waits/recovery and Task executions. Retained. |
| `revision` | Control | Logical execution CAS version. Retained. |
| `event_seq` | Control | Execution event publication cursor, not Step event count. Retained. |
| `agent_run_seq` | Control | Execution-assigned run sequence, not a copied run registration. Retained. |
| `error_code` | Control | Runtime failure classification used by the execution state machine; not diagnostic body. Retained. |
| `safe_error_details` | Terminal outcome reference target | Terminal-details reference code is integrated; handoff sharing is integrated from `d3e56ff8`, not validated. |
| `created_at` | Control | Execution admission time, not a run registration timestamp. Retained. |
| `updated_at` | Control | Latest execution control mutation time. Retained. |
| `mode` | Policy | Fixed accepted Runtime configuration; only changes become overrides. Integrated, not validated. |
| `planning` | Policy | Same shared policy owner. Integrated, not validated. |
| `thinking` | Policy | Same shared policy owner. Integrated, not validated. |
| `binding` | Fixed contract reference | Complete admitted Agent/Task contract has one content owner. No repeated ID/revision/capability/digest summary. Integrated, not validated. |
| `principal_id` | Control | Principal that authorized this execution; Step run metadata is not authorization. Retained. |
| `principal_kind` | Control | Principal category required to interpret authorization identity. Retained. |
| `stored_user_input` | Fixed Step payload or exact Native input location | All sizes use the accepted payload owner; eligible exact Native content shares that source without duplicate digest/size. Integrated, not validated. |
| `producer_generation` | Control | Monotonic producer fencing, including before any Step publication. Retained. |
| `producer_claim_id` | Control | Identifies the current producer claim. Retained. |
| `parent_invocation_id` | Fixed invocation identity | Connects subagent execution to its admitted invocation. Retained. |
| `memory_scope` | Request-specific scope | Per-request routing selection, not a Runtime default snapshot. Retained; absent value omitted by policy implementation. |
| `conversation_agent_run_id` | Fixed Step run identity | Selects the run used for conversation association without copying run facts. Retained. |
| `result` | Terminal outcome reference target | Entire ResultRecord template reference is integrated, with output using its fixed payload owner. |
| `repository_instructions` | Fixed instruction content / retained Asset references | Accepted instructions use immutable content; eligible Asset definitions remain fixed-version references. Workspace/custom content without a retained equivalent owner is preserved. Existing reference implementation. |
| `input_context` | Fixed admitted context reference | Captured context selection belongs to its accepted owner, not latest mutable history. Existing reference implementation. |
| `context_imported` | Control | Records the execution's import decision, not the context body. Retained. |
| `error_diagnostics` | Terminal outcome reference target | Terminal-details reference is integrated; cross-template error sharing is integrated from `d3e56ff8`, not validated. |
| `correlation` | Request-specific metadata | Execution admission correlation, not universally equivalent to Step metadata. Retained; no claim of a second Step summary. |
| `task_attempt` | Control | Task retry coordination; Task executions need not create Step runs. Retained. |
| `task_deadline_at` | Control | Runtime Task deadline. Retained. |
| `task_next_attempt_at` | Control | Runtime retry scheduling. Retained. |
| `dependency_hold_ids` | Fixed retention-hold identities | Identifies independently managed holds; does not duplicate held content. Retained. |
| `requires_task_invocation_capture` | Control | Admission provenance survives failed capture storage or released holds. Retained. |
| `retention_closed` | Control | Monotonic execution publication proof, including relevant conversation receipt. Retained. |
| `started_at` | Control | Execution start, not necessarily any individual Step run start. Retained. |
| `budget_scope_id` | Fixed Runtime budget identity | Execution's admitted budget scope; counters are owned elsewhere. Retained. |
| `tool_response_ref` | Fixed AssetVersionRef | Per-request fixed Asset selection; never resolve the current Asset head. Retained; absent value omitted by policy implementation. |
| `tool_effects` | Policy | Fixed accepted Runtime configuration plus actual override. Integrated, not validated. |
| `session_context` | Policy | Same policy owner; does not reproduce conversation content. Integrated, not validated. |
| `native_step_source` | Fixed Step source | Store/group identity needed to resolve Native checkpoints, not checkpoint contents. Retained. |
| `restart_checkpoints` | Policy | Accepted retention guarantee; not inferred from whichever checkpoints currently exist. Integrated, not validated. |
| `durable_trace` | Policy | Accepted trace guarantee. Integrated, not validated. |
| `conversation` | Policy | Accepted conversation retention guarantee. Integrated, not validated. |
| `recovery` | Control plus fixed recovery content references | Whole pending/handoff references are integrated; nested argument and error-sharing completion is integrated from `d3e56ff8`; see below. |
| `history_seal_digest` | Fixed seal identity | Identifies the execution's accepted history closure; not a duplicate body summary. Retained. |
| `completed_at` | Control | Immutable execution completion boundary, distinct from later control updates. Retained. |

## Nested recovery and outcome boundaries

| Nested value | Ownership boundary | Status |
| --- | --- | --- |
| `recovery.agent_run_id`, `source_agent_run_id` | Fixed run identities selected by recovery | Retained identifiers, not run copies. |
| `recovery.pending_tools` | Entire pending continuation owner | Complete template reference is integrated. Nested argument references to the shared Step payload owner are integrated from `d3e56ff8`, not validated. |
| `recovery.repository_instruction_overlay` | Fixed instruction content with retained Asset sources where applicable | Existing reference implementation; preserve exact accepted semantics. |
| `recovery.repository_instruction_barriers` | Runtime observation/approval frontier | Keep authoritative frontier facts unless the new fixed owner explicitly includes them; never recompute from current Asset heads. |
| `recovery.handoff_phase` | Runtime recovery state-machine coordination | Retained control. |
| `recovery.terminal_handoff` | Fixed terminal handoff/outcome content plus necessary phase control | Complete handoff template reference is integrated. Error sharing with terminal_details is integrated from `d3e56ff8`, not validated. |
| `result.output` and handoff outcome output | Step payload or verified exact Native location | All-size fixed payload references and the enclosing outcome template are integrated; no verification result is asserted. |
| `result.stop_reason`, `usage`, `created_at` | Entire accepted execution outcome | Included in the integrated outcome template owner; not reconstructed from an arbitrary last Step event or model response. |

## Physical kind boundaries

- `tool_operation`: one Step ledger owner for the effect state machine, with
  arguments/result/error pointing to the shared immutable Step payload owner.
  It does not retain a second payload body inside the ledger.
- `agent_run`: Step admission relation holds domain membership and actual
  projection differences; canonical registration remains in `ai_step_runs`.
  The relation is not a second RunRecord body.
- `execution`: retains its own control and fixed references as mapped above.
  Hydrated API objects may contain full values without persisting those copies.
- Captures remain independent owners because source executions can be removed.
  Their binding references reuse the contract owner; they must not borrow the
  source execution's lifetime. Runtime configuration content is not described
  as Step content merely because both are referenced.

Built-in SQL and Memory routing implement these owners. A third-party StateStore
owns its physical layout; preserving its interface does not prove that its
external storage has migrated. Portable logical records are a transport format,
not permission to restore built-in owners into the generic records table.

## Completion boundary

The integrated first version covers the owner/reference changes listed above.
Pending-template argument extraction and the handoff/terminal-details error
sharing are integrated from `d3e56ff8` and not yet validated. This document does not equate moving an outer
template with eliminating every nested duplicate. Small-value thresholds,
reconstructed binding summaries, and moving only output bytes are not substitutes
for the whole-owner requirement.

After publication, CI and targeted evidence must establish exact recovery,
source identity, same-transaction publication, split-group behavior, snapshots,
last-owner collection, and actual SQL/space cost at the combined head. Those
checks are deliberately not reported as completed here.
