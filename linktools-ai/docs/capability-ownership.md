# Capability composition, binding and restore ownership

The complete [capability composition program](../examples/capability_composition.py)
uses the existing public APIs, an offline model, real local tools and temporary
storage. It complements [domain composition](domain-composition.md) and the
[complete MCP example](mcp-protocol-owner.md). Run it after installing through
the repository entry point:

```sh
python manage.py install --editable
python linktools-ai/examples/capability_composition.py
```

The application code is also the retained call shape for internal ownership
changes: no extra manager, producer, adapter, parameter or context is proposed
for these ordinary paths. This is not a claim that all internal ownership work
is complete. In particular, the Task integration decisions below remain open.

## Declaration to execution

1. `RuntimeFeatures` holds host registrations. `capability(value, *, id=None,
   revision=1, config=None, scope="runtime")` accepts native SDK capabilities,
   including native combined and wrapper capabilities. Agent scope is explicit.
   Runtime scope is current host policy, not another Agent-selected contract.
2. `await features.resolve()` produces an inspection-only
   `RuntimeFeaturesResolution`. `AssetCapability` supplies current declarations
   and version references for that call. Pass the original declaration to Runtime
   to transfer Asset ownership, or `features.borrow()` to share a live source
   whose external owner initializes and closes its store. Every kind and loader
   slot remains current; prior resolutions do not freeze later reads.
3. `AgentCompiler.compile` applies Agent selection and authoring defaults.
   `bind` creates the resolved `AgentBindingContract`, including admitted native hook
   references (`kind`, `id`, `revision`). It is the execution
   meaning used for admission, not a second mutable declaration catalog.
4. Runtime selects the execution's principal, captured inputs, Memory scope,
   resource versions and enabled persistence guarantees. It binds native
   run-local services and tool wrappers. This composition must preserve the SDK
   effective run instance, wrapper hooks and ordering.
5. Restore validates saved contracts and materializes their retained meaning.
   It does not rewrite retained tool results or effect receipts. New model steps
   and calls apply the admitted Agent selectors and current filter to today's
   business tool, MCP and Skill source. Native hooks resolve for each new run.
   Current Runtime policy and admitted effects have different owners and lifetimes.
6. The producer drains and settles its execution before releasing owned
   resources. Closing Task, an observer or a metric recorder is not equivalent
   to closing the Agent producer or its borrowed Persistence.


Opened Workspace roots and permission policies, AssetStore owners and Execution
write buffers remain host resources. Removing their capability declaration does
not close an in-use resource; replacing a physical owner on a running Runtime is
rejected rather than silently using the previous resource. Tool, MCP, Skill and
native hook declarations remain live within those resource boundaries.


## Responsibility matrix

Paths below are relative to `linktools-ai/src/linktools/ai`. They describe
existing owners and the boundary to retain, rather than prescribing file moves.

| Area | Declaration / resolution | Binding and execution | Durable fact / close owner | Remaining boundary work |
| --- | --- | --- | --- | --- |
| Asset / Skill | `capability/_asset.py`, `_declaration.py`, `_skill_source.py` resolve current declarations and selected Asset versions; `agent/_compiler.py` owns selectors | `capability/_skill.py` discovers current allowed Skills and reads each selected version; Workspace mounts are prepared at run start | Runtime initializes and closes the store declared by `AssetCapability(store)`; a borrowed live source keeps its store under the external owner | Keep current reads separate from retained tool observations; do not add a second frozen-resource lifecycle or expand sandbox access during discovery |
| FileSystem / Shell | `capability/_filesystem.py`, `_shell.py` supply tool declarations; Runtime captures Workspace and selected names | Capabilities own tool arguments/results and retry feedback; `runtime/_agent_executor.py` binds the shared Boundary and backend | Workspace owns logical paths, physical deployment, process trees and quiescent close; execution owns authorization and effect facts | Audit contribution/binding adapters together; do not move OS isolation or effect authority into a tool capability |
| Memory | `capability/_memory.py` declares native tools; Runtime captures the host scope | Capability owns model tools and SDK adaptation; `runtime/_memory.py` adapts admitted scope, versions, capture/seed and mutation receipts | Persistence owns record/object transactions; the execution retains its admitted Memory scope; current declarations determine available tools | Keep admitted scope separate from current tool discovery; do not resurrect removed tools from historical defaults |
| Planning | Agent/invocation `planning` enables the tool separately from `allow_tools` | `capability/planning.py` owns the supported `write_plan` surface; `runtime/_harness.py` translates the native store contract | `runtime/_plan.py` owns the session/execution plan and optimistic transaction; `_local.py` selects its owner | Co-locate capability-specific projection only after defining its public store boundary; do not confuse model plans, execution plan mode and Task DAGs |
| MCP | `MCPServerSpec` resolves from the current source under admitted Agent selectors; Asset owns referenced resources | `capability/mcp.py` owns discovery, schema feedback, protocol errors and client cleanup; `runtime/_mcp.py` owns model names, original-name routing, selectors, fixture and Boundary wiring | Native protocol client owns connection cleanup; Workspace owns transport processes; execution owns budget, authorization and effect records | Keep this division; moving model names alone does not justify another public API or registry |
| Subagent | Agent compilation pins selected child contracts | `capability/_subagent.py` exposes tools through the existing delegate; `runtime/_subagent.py` owns child start/wait/cancel and inheritance | The execution producer owns parent/child identity, recovery, budgets and drain | Preserve the narrow delegate; capability must not become another execution engine |
| Execution / Checkpoint / Trace | `capability/_execution.py` and `_persistence.py` declare Conversation and Trace guarantees; `StepPersistenceCapability` selects the native StepStore, while official `StepPersistence` supplies capture and recovery hooks | Runtime binds run-local Conversation and Trace services and adapts the built-in StepPersistence to short atomic node boundaries, with a fresh native run ID per attempt; `AgentRunCapture` supplies shared recorder context | Conversation and Trace persistence own their records and commits; Runtime creates and closes the default StepStore adapter over Persistence's Execution SQL context, while an explicitly supplied StepStore remains caller-owned. StepStore persists native snapshots; execution owns fencing, tool effects, checkpoint references and terminal confirmation | Capability hooks may be grouped; do not add a second checkpoint writer or recovery authority |
| Session | `capability/_session.py` is a host declaration, enabled by default | Runtime consumes the declaration; `SessionService` owns session create/resume and admission | Conversation owns session metadata/context and its durability | Retain the declaration/service distinction; Session is not another model tool or a second history store |
| ReferencedInputs / ToolResults / RunCallbacks / PydanticToolControl | Native capabilities participate in explicit Agent selection or current host policy as appropriate | Existing capability classes own their SDK hooks; execution supplies context and the shared tool boundary | Inputs, result objects, output retention and the current run keep their own lifetime | Keep open native/custom extension, effective run instances and callback scope; do not replace them with an internal-type allowlist |
| ToolResponses | Host fixture is declared once at Runtime scope | Execution validates admitted definitions and replays recorded responses without a live MCP client or live result callbacks | Asset owns fixture bytes and versions; execution saves the admitted reference, while the fixture borrows its reader | Keep fixture validation separate from protocol discovery and live external effects |
| Custom persistent capability | `RuntimeFeatures.capability` records its id/revision and commit declaration | `CapabilityPersistence.bind_persistence` creates a run-local binding; Persistence supplies the writer | Persistence owns atomic callback transactions, receipts and configured partitions; host owns borrowed buffers | This is not a cross-worker Python callback queue; do not infer durable replay of uncommitted owner-local callbacks |
| Compaction / repository instructions | Current rules and explicit compaction policy feed execution | Context projection and model-request hooks remain separate from provider identity and raw history | Execution owns requests, usage, budgets and retained history; Asset owns source rules | Clarify capability projection separately from authoritative model facts; no observation-layer dependency on an SDK producer |
| Metrics / Evaluation / Task | Metrics is independently usable; Evaluation declares Task targets; Task owns DAG declarations | Evaluation reuses Task; an Agent node calls the original Agent producer | Each producer owns its recorder; Task owns scheduling/graph recovery; Persistence owns shared transactions | Optional metrics cannot change execution truth; ordinary Task and Evaluation must not activate an Agent producer |

## Public calls exercised by the program

| Call | Meaning, defaults and result | Ownership / failure |
| --- | --- | --- |
| `AssetCapability(store)` | Declare the Asset source in `RuntimeFeatures` | Runtime initializes and closes the store. For an externally shared source, its owner initializes the store before use and closes it after every borrower finishes |
| `features.resolve()` / `features.borrow()` | Resolve returns inspection-only `RuntimeFeaturesResolution`; borrow returns a live `RuntimeFeaturesSource` accepted by Runtime | Each resolution reads current registrations and loader slots with per-read Asset integrity. Borrowing transfers no resource ownership; a source revision change does not invalidate it |
| `FileSystem(workspace).prefix_tools("files")` | Native SDK wrapping changes model-visible names, not logical Workspace path or original tool policy | Runtime binds the backend and closes its owned resources; the declaration is reusable |
| `MemoryCapability(scope="default")` | Host declaration supplies Memory tools for ordinary new runs; selected tools still follow the Agent's tool policy | Runtime resolves the admitted scope. Per-call `memory_scope` is not a user parameter |
| `ExecutionCapability(retain_history=True)` | Retains committed model context used by retries and later turns | It does not enable all Checkpoint/Trace guarantees implicitly |
| `StepPersistence(store=steps, capture_frontier=True)` / `TraceCapability()` | Independent native checkpoint and execution-evidence declarations; frontier capture also saves accepted input and tool calls before dispatch | Runtime borrows the caller's StepStore. A trace does not grant interrupted-run recovery; checkpoint-enabled successful execution alone does not demonstrate crash recovery |
| `SqliteStepStore(database=path)` | Official file-backed StepStore, separate from Runtime's control and Trace storage | Opens and closes short-lived connections per operation; it has no store-level close/context API. With `connection=...` instead, the caller must keep that connection open until Runtime closes, then close it |
| `agent.session(session_id).run(prompt, planning=True, timeout_seconds=...)` | Returns `WaitResult[ExecutionResult]`; planning enables `write_plan` even with empty ordinary tool selection | Session is a handle, not a new resource context. A wait timeout does not terminate the producer |
| `runtime.executions.get(execution_id)` then `source.retry(prompt, idempotency_key=...)` | Gets the retained execution and starts a retry using its admitted semantics | Reopen must provide the required current definitions/resources. The example demonstrates retained Memory, not arbitrary removal of every capability |
| `execution.transcript(include_content=True)` / `execution.model_interactions()` | Read producer-owned history and model facts | Trace/retention policy still determines what exists; observers do not own the producer |

The resource scenario actually invokes `load_skill`, wrapped `list_files` and a
local `run_command`. It changes an authoring resource only after Runtime has
opened, then proves that an earlier resolution retains old bytes while a new
resolution and Skill calls see current bytes.
The native host hook remains active even when Agent-selected capabilities are
empty. The Memory scenario writes real state, closes Runtime, opens a host with
no Memory declaration and checks a fresh run has no Memory tools. It then adds
Memory to that same live source and retries the original execution using current
tools and the original admitted scope, rather than the new host default. The Planning scenario
uses the real native tool and reads Session/Trace outputs through public APIs.
Its caller-owned SQLite StepStore remains available after Runtime closes, and
the example verifies that the official run and committed snapshot still exist.
The temporary directory outlives both Runtime and these final store reads.

These examples do not exercise a killed worker, unknown external-effect
outcome, rejected authorization, cleanup retry, Subagent recovery, custom
persistent callback recovery or every native extension hook. Their regression
tests remain necessary; a successful example does not replace them.

## Task to Agent: complete current integration surface

`AgentExecutionPort` is exported from `linktools.ai.runtime.agent_execution`
and reachable as `Agent.execution_port`. Its dataclass fields and overridable
methods are a public surface. Task currently consumes the following paths:

| Consumer path | Current operation / result | Contract that a narrower boundary must preserve |
| --- | --- | --- |
| `Task.from_agent`; `AgentTaskRunner` construction | Read `compiled.spec` and `binding_contract` | Capture declaration identity, planning/thinking defaults and the full binding once; do not recapture per operation |
| `TaskEngine` and runner binding | Exact `persistence` identity; `operations.ensure_open()` | A matching database URL does not replace shared ownership; reject a closed producer before work |
| Admission input | `canonicalize_files(files)` → tuple; `materialize_input(prompt, files=())` → canonical input | Canonical Workspace-relative paths, original tenant and the existing captured content boundary |
| Durable binding validation | `resolve_contract(contract)` → `AgentBindingContract` | Resolve the saved selection, not current wildcard expansion; reject contract drift |
| Start and get | Overridable `start(...)` / `get(execution_id, principal)` → `Execution` | Original principal, output contract, correlation, graph budget scope, invocation-capture requirement and dependency hold |
| Invocation capture | Optional `captures.record_invocation(execution_id, invocation)` | Persist Task-to-execution capture before releasing the handoff obligation |
| Authorized reads | `service.inspect`, `result`, `result_payload_size`, `wait`, `list_children`, `budget_usage` | Existing authorization and exact producer service. Service `wait` returns `ExecutionResult`, not the observer's `WaitResult` |
| History and live observation | `operations.history` cutoff/read/subscribe operations; `watch_tree` | Root/child producer routing, cursors, readiness, stream cleanup and observation lifetime |
| Task execution handle | `operations`, `watch_tree` plus Task-owned cancel callback | Node cancellation must also settle the graph. Returning ordinary `port.get()` alone loses that callback |
| Cancellation | `service.cancel(execution_id, CancelExecutionRequest(principal, key, force))` → `CancelExecutionResult` | Graph settlement distinguishes confirmed, unconfirmed and unknown effects; observer cancellation does not imply producer cancellation |
| Retention | `service.acquire_dependency_hold(..., tenant_id, hold_id)` → bool; release → None | The principal-based `port.acquire_hold(...)` returns None and is independently overridable; it is not an equivalent replacement |
| Terminal handoff | `service.request_terminal_handoff(..., tenant_id, terminal_commit=False)` → None | Forward the terminal flag and keep existing holds/readback. A missing producer only permits already-settled, durable no-op cleanup under the existing checks |
| Recovery | `owns_execution(execution_id, *, tenant_id)` → bool; `recover_bound_execution(record, *, recovery_operation)` → `ExecutionRecord` | Task validates the graph/node binding and recovery actor; Agent resumes the original execution identity under current fencing |

The existing regression in `test_agent_task_binding_owner.py` deliberately
allows `port.service` and `port.operations.service` to differ. Start/get overrides
also remain independent from service dispatch. A blanket equality requirement,
replacement with a handle, or use of `acquire_hold` would change this contract.

### Decision required before changing the public port

The next boundary design must specify whether it retains existing public
collaborators with narrower protocol types, or replaces direct access with
producer operations and an explicit transition for existing port implementers.
It must cover the whole table, including history, graph-aware cancellation and
tenant-level retention. Merely adding wrappers for start/get/inspect leaves the
rest unchanged and adds another maintenance surface.

No new port names, parameters, default values, constructor shape or lifecycle
are adopted by this document. Ordinary `Task.from_agent` and Runtime composition
remain unchanged in either design. Runtime's current execution/turn authority
and Persistence contracts are not part of this API decision.
