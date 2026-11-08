
# Tool identity and explicit Trace visibility

An ordinary `RuntimeFeatures.tool` receives an invocation-local `AgentContext`.
Its `tool_call` is an immutable `ToolCallIdentity` containing the admitted principal,
namespace, execution ancestry, session, agent and run sequence, plus the native
run ID, tool-call ID/name and retry count. It does not query history. These two
invocation fields, `tool_call` and `trace`, are absent outside a bound tool call.
Native SDK tools and capability hooks can instead use
`ToolCallIdentity.from_context(ctx)` and `TraceCapability.for_context(ctx)`.

`model_request_seq` comes from the existing captured call provenance, including
restored calls. It is `None` when Trace does not establish that fact; `run_step`
and `retry` are not substitutes. Identity is correlation data, not a grant of
authority: application authorization still checks the admitted principal and
relevant resource policy.

`await context.trace.flush()` (ordinary tools) or
`await TraceCapability.for_context(ctx).flush()` (native hooks) explicitly publishes
the current run's already-captured Trace frontier. Successful return means the
source archive has confirmed publication, not merely that a write buffer accepted
data. A transient route without a source archive rejects this operation. Other
runs are not drained, future observations retain the selected batch policy, and
business/checkpoint state is not committed by this call. Cancellation and source
failures follow the existing publication settlement rules and never report a
successful barrier before confirmation.

Batch and terminal modes do not promise that a separate reader can see the current
call before this barrier. Prefer the direct identity when only correlation is
needed. An authorization hook requiring persisted evidence can explicitly flush
after that evidence has been captured. A flush does not synthesize later hooks:
for example, an outer hook running before Trace's tool-start hook cannot flush a
`TOOL_CALL_STARTED` event that has not happened yet.

## Runtime boundary regression coverage

The upstream integration tests exercise these contracts with SQLite and both
batch and terminal Trace policies:

- A real root execution delegates to a child execution whose ordinary tool
  receives the child's execution ID, root/parent IDs, admitted principal and
  model-request provenance. An independently opened persistence reader sees no
  current call before the explicit barrier and sees the matching call afterward.
- A worker process exits abruptly after the first tool's explicit publication
  and checkpointed result. A fresh process reopens the Runtime, preserves the
  original execution/session/principal and captured Trace facts, and resumes
  with a new agent-run identity. The next tool's request sequence belongs to that
  new run, rather than being interpreted as an execution-global counter.
- Recovery does not repeat the first tool. Original source events remain intact,
  and the later call retains its own explicit publication boundary and provenance.

These tests use a deterministic model, the same authorized actor after restart,
and the currently supported single level of subagent delegation. They do not
claim recursive subagent delegation, a different recovery actor, remote model
provider behavior, or downstream application authorization acceptance.
