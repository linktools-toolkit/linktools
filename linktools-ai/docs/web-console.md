# Local Web console

The console adds a local browser interface to the same Runtime used by the AI
commands. The CLI now contains only `ai-run`, `ai-acp`, and `ai-web`; the five
inspection commands are consolidated here. Runtime still owns the execution
store, conversation history, scheduling, recovery, and metrics.

## Start

```bash
python manage.py install --editable 'linktools-ai[web]'
ai-web --project /path/to/project --model your-model
# Equivalent entry point:
python -m linktools ai web --project /path/to/project --model your-model
```

The execution stack requires `pydantic-ai-slim>=2.53.0,<3.0.0`. After updating
the checkout, rerun the install command in the same virtual environment as
`ai-web`; an existing editable install does not refresh dependency metadata
until it is installed again. `--help` only exercises lightweight CLI discovery.

Open `http://127.0.0.1:8765`. `--port` selects another local port, and `--open`
opens the URL in your browser. The existing `--base-url`, `--api-key`, `--vision`
and `--memory` arguments have the same meaning as `ai-run`; prefer environment
configuration for credentials rather than shell history. The console does not
persist or edit model credentials.

Shared CLI configuration uses the `LINKTOOLS_` environment prefix. For example,
`LINKTOOLS_OPENAI_VISION=true` supplies the typed vision setting used by both
`ai-run` and `ai-web`; the console displays that resolved setting.

```bash
ai-web --project /path/to/project --read-only
```

Read-only mode opens `RuntimeHistory`, never an execution `Runtime`, and does not
invoke recovery. It also works without a model configuration. An unconfigured
model automatically selects this mode. Missing history and metrics stores are
shown as empty/unavailable, without provisioning them through a history read.

### Finding details and controls

The default conversation view keeps messages and the session's current state
in focus. Session occupancy comes from its active execution identity; inspecting
an older turn does not substitute that turn's status for the session's owner.
**View active execution** returns to the owner when a different turn is selected.

- **Details** opens session metadata and the selected execution's Overview,
  History, Transcript, Models & prompt, Trace, and Recovery panels. Usage,
  full JSON, result export, parent/subagent links, and detail paging stay here.
  Inspecting a turn or opening an exact execution reveals the panel. Closing it
  returns focus to Details without stopping observation or changing selection.
- **Actions** groups conversation rename/fork/close and execution retry/fork/end.
  Stop remains directly available outside the collapsed panel. Resume and
  external-effect decisions remain in Recovery; read-only mode disables writes.
- **Message options** contains planning, thinking, memory scope and attachments.
  The send shortcut and earlier-turn paging remain available in the main view.
- **Execution filters** and **Open exact ID** expand in the sidebar; applied
  filters, explicit newest-20 scans and list paging retain their original scope.
  Metrics and Runtime settings keep their sidebar entries.
- Tool/thinking content and ordinary live activity expand on demand. Failure,
  recovery, cancellation and input-required events remain visible, as do request
  errors and observation interruptions. A healthy connection is not presented as
  a second execution status.

### Shared storage and explicit recovery

The default local SQL Runtime can open alongside another process using the same
namespace, tenant, and storage without resuming its live execution. Writable
and read-only consoles read the same committed history. Independent sessions
can run concurrently; a second execution in the same session is rejected with
`SESSION_BUSY`.

SQL startup does not infer that an unfinished execution is abandoned. A stopped
process can leave `PENDING_START`, `STARTED`, or `CANCELLING` visible. Refresh and
observation never take ownership. After confirming the previous executor has
stopped, use **End stopped execution** to cancel the old work and release its
session in one action. The console reads the durable cancellation state before
finishing it through Runtime recovery, then verifies a terminal execution and
released session owner. It does not resume model or tool work. Unknown outcomes
are not reported as complete; another explicit attempt reads the current state
before continuing. Resolve unknown external effects in **Recovery** if needed.

To continue the original work instead, **Recovery → Resume stopped execution**
retains the explicit stopped-executor confirmation and may call models or tools.
**Stop execution** remains available for a running executor. A cancellation
request may remain `CANCELLING` until the owner finishes or the stopped-execution
action completes its cancellation.

Runtime owns producer fencing, terminal handoff, and local staging release.
The console does not introduce a private lock, heartbeat, background result
consumer, or alternative recovery implementation. See the
[Runtime history guide](runtime-history.md) for storage-specific recovery and
publication guarantees.

## Command consolidation and coverage

`ai-history`, `ai-session`, `ai-status`, `ai-trace`, and `ai-metrics`, including
their `lt ai` / `python -m linktools ai` forms, are intentionally removed. There
are no compatibility aliases. Start `ai-web --read-only` to inspect an existing
workspace, then choose Session or Execution in **Open exact ID** to open a known
identity. `ai-run` and `ai-acp` remain available for terminal execution and ACP
stdio clients. Runtime library APIs are unchanged.

The former viewers had no filter or export flags. Their optional session or
execution IDs, required trace execution ID, and optional metric name map to the
controls below. `ai-run --json` remains available, alongside final-result JSON
download in the execution inspector. Full public records can be expanded in
place; diagnostic exception messages remain redacted to protect credentials.

| Command capability | Console surface | Runtime source |
| --- | --- | --- |
| `ai-run` | New conversation, composer, planning/thinking/memory/files, live text and tool/subagent progress, Stop, final JSON download | Agent/Session `start`, Execution `watch`/`cancel`, RuntimeHistory `result` |
| Former `ai-session` | Session list, status/revision/CWD/active execution, paged turns, create/rename/fork/close | RuntimeSessions and RuntimeHistory session APIs |
| Former `ai-history` | Paged executions, explicit newest-by-time scan, metadata, complete paged history/transcript, seven-layer prompt architecture, model request/response/duration/cache usage | RuntimeHistory `list_executions`, `recent_executions`, `inspect_execution`, `history`, `transcript`, `model_interactions` |
| Former `ai-trace` | Metadata timeline with source execution, run/request/step/tool locators, duration, purpose, token usage and content drill-down | RuntimeHistory `trace` and filtered `history` |
| Former `ai-metrics` | Twelve-metric summary, named metric, time window, dimensions/correlation filters, grouping, aggregation and buckets | Metrics `query` |
| Former `ai-status` | Runtime & capabilities: workspace/storage locations, model, vision, credential presence, captured Agent/Tool/Skill/MCP identities | CLI composition and one CapabilityGroup capture |
| `ai-acp` | Equivalent explicit session create/load/continue/fork/close/cancel interactions; ACP itself remains a separate stdio transport | The same public Session and Execution APIs |

Retry, fork, recovery and external-effect resolution are explicit actions in the
execution inspector. Recovery uses the Runtime's actual operation ID and fence;
the console never assumes whether an external effect was applied. Selecting a
child execution preserves its identity rather than treating its events as a
root model call. Full JSON details preserve public fields that do not fit the
compact presentation.

The **Models & prompt** tab summarizes system/fixed/dynamic instructions,
conversation context, attachment media types, tools and output contracts from
the selected recorded request. It does not duplicate the instruction mirror or
fetch trace content in the background. Full prompts/responses remain expandable.

List filters follow public API semantics. Execution queries support session,
agent and parent identity filters. The text search filters only loaded rows;
it is not a global full-text search. Cursor pages are passed through unchanged.
Normal execution browsing uses repository cursor order. To reproduce the former
exact newest list, select **Newest 20 (metadata scan)** under Execution order and
apply it explicitly. The current public API scans visible metadata in O(N); this
mode has no cursor or server filters and is never triggered by refresh/polling.
To inspect all executions for a session (including non-conversation children),
use its Session ID filter in Executions rather than only the conversation turns.

Read-only `RuntimeHistory` currently exposes a recent-session list rather than a
paged session index; the UI labels this limitation. Writable mode uses the
paged RuntimeSessions API. Browsing an exact session still pages its turns. If
the Runtime reports unavailable session history, the console reads metadata
separately and shows the history error explicitly. Status, revision, working
directory and active execution remain inspectable without pretending that the
conversation is empty.

## Observation and ownership

The CLI opens resources once, captures CapabilityGroup once for both inspection
and Runtime composition, runs one HTTP server, and closes resources after the
server stops. `create_app` accepts already-open public Runtime/RuntimeHistory and
Metrics objects for embedding and testing; the caller owns their lifespan.

Sending a message returns its execution ID independently of the SSE connection.
A browser disconnect closes its observer, not the execution. Stop requests
cancellation and reads the actual Runtime outcome; acceptance does not imply
that external effects or resource cleanup have finished. A concurrent Runtime
update can return a cancellation conflict. The console rereads the authoritative
status and lets you explicitly choose Stop again with the same request identity;
it never treats a conflict as success or retries cancellation automatically.
Server shutdown uses
Runtime's normal close behavior rather than creating browser-owned executions.

SSE passes through Runtime watch cursors and metadata-only model projections.
Model rows are replaced by `(execution_id, agent_run_seq, model_request_seq)`;
usage is never added a second time on replay. Terminal request metadata is not
replaced by a stale RUNNING projection. Transient response text is discarded on
a broken stream and reconciled from authoritative history before reconnecting,
rather than appended twice. A finished stream is followed by an authoritative
execution read. Cursor/authorization/integrity failures do not silently become
success or restart an execution.

History, trace and model requests load only for the selected execution/tab.
Tool arguments/results and model prompt/response bodies are not stored in a
second Web cache or sent through Redis. Retained live progress can precede
committed history; the UI explicitly shows incomplete or pending content. Independently opened readers see the committed
prefix after Runtime publishes its observation batch, including RUNNING model
identities without request/response bodies. Completed requests and controlled
abnormal ends publish their available content. Refresh the same exact identity
to see newer facts. A live notification or empty page is not evidence of a
provider outcome. Publication, cross-process visibility, and recovery remain
Runtime responsibilities.

## Local trust boundary

- The CLI binds only `127.0.0.1`; no remote-host option is provided
- The HTTP adapter checks the loopback peer, exact local Host/port, browser Origin
  and cross-site fetch metadata; mutation requests require JSON and a custom
  same-origin header, with no permissive CORS policy
- Forwarded proxy headers are disabled; the actual connection peer must be
  loopback, including in proxy mode
- No persistent access token, login system or new permission model is added
- Credentials are represented only as configured/missing; exception messages
  are omitted from HTTP diagnostics because they can contain provider secrets;
  error code, safe details, exception type and cause digest remain available
- Model/user/tool content is rendered as text, not executable HTML; static assets
  are bundled locally, and CSP disallows external scripts, frames and plugins

An HTTP JSON request is bounded at 8 MiB, including envelope/escaping. Runtime
prompt, file, tool-result and authorization validation still applies. No browser
path bypasses Workspace policy.

### Reverse proxy

For a reverse proxy on the same machine, opt in explicitly:

```bash
ai-web --proxy --project /path/to/project --model your-model
```

`--proxy` accepts the proxy's external Host and browser Origin, including HTTPS
origins. It does not change the loopback listener or trust forwarded peer
headers. The proxy must connect to `127.0.0.1` and provide authentication, TLS,
and access control. The console has no login: an unauthenticated public proxy
exposes history and execution, cancellation, and recovery controls.

Cross-site fetch rejection, the mutation request header, and the absence of
permissive CORS remain in force. The default mode retains local Host/Origin
validation. Serve the UI and API at the same external origin.

## Verification

Run the repository gate with the optional Web dependencies available:

```bash
python manage.py check linktools-ai
python manage.py build linktools-ai
python manage.py verify linktools-ai
```

Session identities travel in query parameters for detail/control requests, so
opaque IDs containing slashes, Unicode, query delimiters or dot segments keep
their exact Runtime meaning.

The Web regression tests cover the local-origin boundary, remote-peer rejection,
secret-safe diagnostics, real session/execution idempotency, history cursor
paging and content opt-in, trace/model/result/metric parity, session revision
conflicts, fork/close, read-only reopen without models, SSE cursor replay and
observer disconnect without cancellation. Cross-process HTTP tests also cover
live reads without startup takeover, explicit recovery after process exit, and
orphaned cancellation without restarting the provider. A controlled SQL revision
conflict verifies canonical readback and explicit same-identity retry without
duplicate cancellation events. A Node-backed client test checks Unicode/chunked
SSE parsing, root/child/request identity, terminal metadata
ordering and duplicate page replacement. A lightweight DOM contract harness
also exercises stale navigation/details/actions, an interrupted creation dialog,
uncertain fork identity, repeated sends and edits to the next draft. Recovery
checks cover all recoverable statuses, refresh without takeover, confirmation
rejection, repeated clicks, and terminal control reconciliation. These are
interaction-logic tests, not browser rendering tests; they skip explicitly if
Node is absent.

Browser acceptance should cover desktop/mobile layout; Unicode and hostile
HTML text; create/send/repeated submit; switching sessions/tabs during pending
requests; reload/reconnect while running; Stop; model failures; child/tool/model
detail navigation; cursor Load more; read-only empty stores; JSON export; and
closing/reopening dialogs. Use fake models and an isolated fixture store.
