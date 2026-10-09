# Local Web console

The console adds a local browser interface to the same Runtime used by the AI
commands. It does not replace the commands or create another execution store,
conversation database, scheduler, recovery coordinator, or metric aggregator.

## Start

```bash
python manage.py install --editable 'linktools-ai[web]'
ai-web --project /path/to/project --model your-model
# Equivalent entry point:
python -m linktools ai web --project /path/to/project --model your-model
```

Open `http://127.0.0.1:8765`. `--port` selects another local port, and `--open`
opens the URL in your browser. The existing `--base-url`, `--api-key`, `--vision`
and `--memory` arguments have the same meaning as `ai-run`; prefer environment
configuration for credentials rather than shell history. The console does not
persist or edit model credentials.

```bash
ai-web --project /path/to/project --read-only
```

Read-only mode opens `RuntimeHistory`, never an execution `Runtime`, and does not
invoke recovery. It also works without a model configuration. An unconfigured
model automatically selects this mode. Missing history and metrics stores are
shown as empty/unavailable, without provisioning them through a history read.

### Runtime prerequisites for long-lived execution

The Runtime version on which this console is based has a verified startup
recovery limitation: opening another writable Runtime with the same
namespace/tenant/storage and compatible bindings while a different process is
still executing can resume that live execution a second time. This is an
upstream startup-recovery issue, not a restriction on independent work in
already-open Runtime instances: distinct sessions can run concurrently, and a
second execution in the same session is rejected with `SESSION_BUSY`.

Use `--read-only` to inspect an already-running CLI process until the upstream
startup ownership fix is available. The console does not introduce a private
lock, owner lease, heartbeat, or alternative recovery implementation.

The same baseline also retains completed execution staging when no owner-side
`Execution.wait`/result consumer performs its terminal handoff. Browser watches
and `RuntimeHistory` queries do not consume that handoff, so a long-lived
writable console can accumulate retained local staging even after durable
completion. This is another upstream Runtime lifetime issue. The console does
not add a background result consumer solely to compensate for it.

Integrate the published Runtime fixes for both startup ownership and terminal
staging release before treating long-lived writable operation as validated.
This console branch uses the published master APIs and does not incorporate
unpublished persistence or lifecycle changes.

## Interface and command coverage

| Existing command | Console surface | Runtime source |
| --- | --- | --- |
| `ai-run` | New conversation, composer, planning/thinking/memory/files, live text and tool/subagent progress, Stop, final JSON download | Agent/Session `start`, Execution `watch`/`cancel`, RuntimeHistory `result` |
| `ai-session` | Session list, status/revision/CWD/active execution, paged turns, create/rename/fork/close | RuntimeSessions and RuntimeHistory session APIs |
| `ai-history` | Executions and inspector: metadata, complete paged history, transcript, model prompts/responses and usage | RuntimeHistory `list_executions`, `inspect_execution`, `history`, `transcript`, `model_interactions` |
| `ai-trace` | Metadata timeline with source execution, run/request/step/tool locators and content drill-down | RuntimeHistory `trace` and filtered `history` |
| `ai-metrics` | Twelve-metric summary, named metric, time window, dimensions/correlation filters, grouping, aggregation and buckets | Metrics `query` |
| `ai-status` | Runtime & capabilities: workspace/storage locations, model, vision, credential presence, captured Agent/Tool/Skill/MCP identities | CLI composition and one CapabilityGroup capture |
| `ai-acp` | Equivalent explicit session create/load/continue/fork/close/cancel interactions; ACP itself remains a separate stdio transport | The same public Session and Execution APIs |

Retry, fork, recovery and external-effect resolution are explicit actions in the
execution inspector. Recovery uses the Runtime's actual operation ID and fence;
the console never assumes whether an external effect was applied. Selecting a
child execution preserves its identity rather than treating its events as a
root model call. Full JSON details preserve public fields that do not fit the
compact presentation.

List filters follow public API semantics. Execution queries support session,
agent and parent identity filters. The text search filters only loaded rows;
it is not a global full-text search. Cursor pages are passed through unchanged.
Read-only `RuntimeHistory` currently exposes a recent-session list rather than a
paged session index; the UI labels this limitation. Writable mode uses the
paged RuntimeSessions API. Browsing an exact session still pages its turns.

## Observation and ownership

The CLI opens resources once, captures CapabilityGroup once for both inspection
and Runtime composition, runs one HTTP server, and closes resources after the
server stops. `create_app` accepts already-open public Runtime/RuntimeHistory and
Metrics objects for embedding and testing; the caller owns their lifespan.

Sending a message returns its execution ID independently of the SSE connection.
A browser disconnect closes its observer, not the execution. Stop requests
cancellation and reads the actual Runtime outcome; acceptance does not imply
that external effects or resource cleanup have finished. Server shutdown uses
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
second Web cache or sent through Redis. Live progress can precede durable
history on the current Runtime; the UI explicitly shows incomplete or pending
content. Incremental persistence, stronger cross-process visibility and exact
recovery behavior remain Runtime responsibilities.

## Local trust boundary

- The CLI binds only `127.0.0.1`; no remote-host option is provided
- The HTTP adapter checks the loopback peer, exact local Host/port, browser Origin
  and cross-site fetch metadata; mutation requests require JSON and a custom
  same-origin header, with no permissive CORS policy
- Proxy headers are disabled; do not publish this local-trust application behind
  a proxy or mount it as a remotely authenticated service
- No persistent access token, login system or new permission model is added
- Credentials are represented only as configured/missing; exception messages
  are omitted from HTTP diagnostics because they can contain provider secrets;
  error code, safe details, exception type and cause digest remain available
- Model/user/tool content is rendered as text, not executable HTML; static assets
  are bundled locally, and CSP disallows external scripts, frames and plugins

An HTTP JSON request is bounded at 8 MiB, including envelope/escaping. Runtime
prompt, file, tool-result and authorization validation still applies. No browser
path bypasses Workspace policy.

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
observer disconnect without cancellation. A Node-backed client test checks
Unicode/chunked SSE parsing, root/child/request identity, terminal metadata
ordering and duplicate page replacement. A lightweight DOM contract harness
also exercises stale navigation/details/actions, an interrupted creation dialog,
uncertain fork identity, repeated sends and edits to the next draft. These are
interaction-logic tests, not browser rendering tests; they skip explicitly if
Node is absent.

Browser acceptance should cover desktop/mobile layout; Unicode and hostile
HTML text; create/send/repeated submit; switching sessions/tabs during pending
requests; reload/reconnect while running; Stop; model failures; child/tool/model
detail navigation; cursor Load more; read-only empty stores; JSON export; and
closing/reopening dialogs. Use fake models and an isolated fixture store.
