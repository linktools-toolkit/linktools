# linktools-ai CLI: local agent runner design

## Goal

`linktools-ai` has a complete agent runtime (`AgentKernel`, `RuntimeAgent`, skill/subagent/MCP
registries, `FileSession`) but no CLI ever wires it end-to-end — there is no `commands/` package,
no `__main__.py`, and no way for a user to run an agent locally. This adds the first such entry
point: an interactive chat REPL, reachable as `lt ai chat`, that drives a `RuntimeAgent` against
an OpenAI-compatible model backend.

## Non-goals (this slice)

- Loading skills/subagents/MCP servers from disk or DB. Registries are constructed empty; the
  agent only gets the built-in `file`/`terminal` tools.
- Any config-file-based credential storage. Model config comes from CLI flags or env vars only.
- Multi-session management UI (listing/deleting sessions). One session per `--session` id,
  created on demand.
- Non-OpenAI-protocol backends (`build_model` only supports `protocol="openai"` today).

## User-facing behavior

Command: `lt ai chat`

Flags:
- `--model` (default env `OPENAI_MODEL`) — model name passed to the backend.
- `--base-url` (default env `OPENAI_BASE_URL`) — OpenAI-compatible endpoint.
- `--api-key` (default env `OPENAI_API_KEY`) — bearer token.
- `--session` (default `"main"`) — session id; conversation history persists under this id
  across separate `lt ai chat` invocations.
- `--workdir` (default: current directory) — directory the agent's `file`/`terminal` tools
  operate in.

Missing `--base-url`/`--api-key` (and no env fallback) is a clear `CommandError`, not a stack
trace.

Once running: a `> ` prompt reads one line at a time, streams the agent's reply to stdout as
text deltas arrive, and prints a one-line indicator when a tool call starts/finishes (e.g.
`[tool: file start]` / `[tool: file end ok]`). `exit`, `quit`, or EOF (Ctrl-D) ends the REPL.
Ctrl-C during a turn cancels that turn only and returns to the prompt; Ctrl-C at an empty
prompt exits.

## Implementation shape

New files under `linktools-ai/src/linktools/commands/ai/`:

- `__init__.py` — sets `__command__ = "ai"`, `__description__ = "AI agent tools"`, following the
  same convention as `linktools-common/src/linktools/commands/common/__init__.py`.
- `chat.py` — the `Command` (subclass of `BaseCommand`), module-level `command = Command()`.

`pyproject.toml` change: add
`commands = {path = "src/linktools/commands", module = "linktools.commands"}`
under `[tool.linktools.scripts]`, alongside the existing `capability` entry — this is the same
convention `linktools-common`/`linktools-mobile` use (not the single-module `cntr` pattern),
because `ai` should surface as a subcommand group under the shared `lt` dispatcher
(`lt ai chat`), not as a standalone executable.

### `Command.run()` flow (wrapped in `asyncio.run`)

1. Resolve `RuntimeModelConfig` from flags with env fallback (`OPENAI_MODEL`, `OPENAI_BASE_URL`,
   `OPENAI_API_KEY`); raise `CommandError` if `base_url` or `api_key` end up unset.
2. `model_registry.register("standard", RuntimeModelConfig(model_type="standard",
   protocol="openai", model=<resolved>, base_url=<resolved>, api_key=<resolved>,
   auth_token=None, timeout_seconds=300, raw={}))`.
3. Construct empty registries — `SkillRegistry()`, `SubagentRegistry()`, `MCPRegistry()` (no
   filesystem paths passed in) — and `await` each `.preload()`.
4. Build an in-code `AgentSpec` (no `agent.md` file needed): `name="ai"`, `model="standard"`,
   `allowed_tools=["file", "terminal"]`, `allowed_skills=[]`, `allowed_subagents=[]`, a short
   default `system_prompt` describing a general-purpose local coding/ops assistant.
5. `kernel = AgentKernel(skill_registry=..., subagent_registry=..., mcp_registry=...)`.
6. `session = FileSession(session_id=<--session>, root=environ.get_data_path("ai", "sessions",
   <--session>, create_parent=True), status_store=InMemorySessionStatusStore())` — history
   persists across invocations for the same session id.
7. `context = kernel.build_context(spec, session, builtin_tool_names=frozenset({"file",
   "terminal"}))`.
8. `agent = RuntimeAgent(spec, session, execution_context=context, workdir=Path(<--workdir or
   cwd>))`.
9. REPL: read a line via `input("> ")` (in a thread via `asyncio.to_thread` so it doesn't block
   the event loop); on `exit`/`quit`/EOF, break. Otherwise `async for event in
   agent.stream({"question": line})`, printing `event["text"]` deltas as they arrive (no
   trailing newline until the turn ends) and a one-line indicator for `tool` events. Wrap the
   per-turn stream in a `try/except asyncio.CancelledError` so Ctrl-C aborts only that turn.

### Error handling

`Command.known_errors` includes `ModelClientUnavailable`, `ModelOutputError`,
`ModelTurnLimitExceeded` (from `linktools.ai.core.model_runtime`) so these surface as clean CLI
errors rather than tracebacks, consistent with how `cntr`'s `Command.known_errors` is set up.

## Testing

- Unit test for the model-config resolution helper (flags win over env, clear error when both
  missing).
- No test drives a real model backend; the REPL loop itself is exercised manually.
