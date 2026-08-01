# linktools-ai

Agent / session / execution runtime built on
[pydantic-ai](https://ai.pydantic.dev/). Consumers declare an `AgentSpec`,
wire a storage backend + a `ModelResolver`, and call `Runtime.run`. An optional
CLI (`linktools.ai.cli`) and TUI (`linktools.ai.cli.tui`) are included for
local runs and inspection.

## Quick start

A tested minimal example lives at
[`examples/minimal_runtime.py`](../examples/minimal_runtime.py). It registers
a model, builds a `Runtime` over `LocalDirectoryStorage`, runs one no-tool
agent, and closes it:

```python
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model import ModelPolicy, ModelRegistry, ModelResolver
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime

registry = ModelRegistry()
registry.register("standard", model=my_model)

storage = LocalDirectoryStorage(root="./data")
await storage.initialize_storage()

runtime = build_runtime(storage=storage, model_resolver=ModelResolver(registry=registry))
async with runtime:
    spec = AgentSpec(
        id="writer",
        name="writer",
        model=ModelPolicy(primary="standard"),
        instructions=PromptSpec(instructions="You are a careful writer."),
    )
    output = await runtime.run(spec, "Write a one-line summary.")
    print(output)
```

`build_runtime(...)` (in `linktools.ai.runtime`) is the sole construction
entry point — it returns a `Runtime` usable as an async context manager.
`Runtime.run` returns the agent's output directly (not a result wrapper).

## Storage

| Backend | Coordination | Use |
|---|---|---|
| `LocalDirectoryStorage` | single-process | local debugging, tests |
| `SqlAlchemyRuntimeStorage` | process or shared-database | production |

`RuntimeRequirements(topology=RuntimeTopology.MULTI_PROCESS)` rejects a
`LocalDirectoryStorage` execution backend at `build_runtime` time — a
multi-worker deployment must use the SQL-first storage. Tools, tasks, memory,
and artifacts are optional capabilities (`storage.tools`, `.tasks`, `.memory`,
`.artifacts`); `RuntimeRequirements` declares which ones a given deployment
needs so a missing store fails at composition time, not mid-run.

### Storage composition

Every domain store (spec, execution, task, tool, artifact) is a
`StorageComposition`: a primary reader + optional ordered fallback layers + a
writer + an optional content cache. The composition owns parallel per-layer
metadata refresh, owner-aware entry merge, effective revision, and
owner-directed `get` / `get_many` with the metadata-miss rule.

Metadata loading uses a single-load REPLACE-or-PATCH protocol
(`StorageMetadataBackend.load_metadata`): one call returns the head revision
plus either the full entry set (REPLACE) or only the diff since the caller's
held revision (PATCH). An injectable `RevisionSource` lets a downstream system
share one revision signal across processes (redis/file cache) so a hot read
loop short-circuits without hitting the database.

### SQL concurrency model

The SQL backends use **optimistic CAS, no pessimistic locks**. Every mutation
is `UPDATE ... WHERE <all previously-read columns still match> ...` with a
`rowcount != 1` conflict check. Monotonic columns (`fence`, `event_sequence`,
`snapshot_revision`) on every CAS WHERE clause prevent ABA. A writer declaring
the `BatchStorageWriter` capability gets atomic `apply_batch(puts, deletes)`
(one transaction, one shared revision); otherwise the composition falls back
to per-op `put` / `delete`.

The `SqlAlchemyDialect` Protocol (`sqlite` / `mysql` / `postgresql`) provides
vendor-neutral `upsert`, `upsert_many`, and `insert_ignore_conflict` primitives
so no store hardcodes vendor SQL.

### Installation extras

The core wheel carries **no** environment-specific DB driver. Install the SQL
kernel on demand:

```bash
pip install linktools-ai[sqlite]      # SQLAlchemy + aiosqlite (dev/test)
pip install linktools-ai[sqlalchemy]  # SQLAlchemy only (bring your own driver)
```

A production deployment using MySQL or PostgreSQL brings its own async driver
(`asyncmy`, `asyncpg`, …) — the core stays backend-neutral.

## Architecture

```text
build_runtime(storage, model_resolver, requirements, dependencies)
  -> AgentCompiler   (resolves ModelPolicy -> ResolvedModel, compiles AgentSpec)
  -> AgentEngine     (drives the agent: model calls, tool calls, trace collection)
  -> ExecutionService (lifecycle: run / resume / cancel, session + snapshot + trace)
  -> ExecutionQueryService (authorized read model over the same storage)
```

`Runtime` (in `linktools.ai.runtime`) is the public facade: `run`, `resume`,
`cancel`, `inspect`, `aclose`. It never exposes a concrete Store or the
compiler/engine directly.

### Approval pause / resume

When a tool call requires approval, the engine pauses the run: the snapshot
and the pending `RunApproval` are persisted atomically, and the run
transitions to `PAUSED`. `ApprovalDecision.ALLOW` keeps it `PAUSED` (resumable);
`ApprovalDecision.DENY` is terminal — the run transitions straight to
`CANCELLED` without executing the tool.

`Runtime.resume(run_id)` restores the **original** spec + identity from the
persisted snapshot — a caller cannot inject a different model, tool list, or
identity on resume.

### Cancel

`Runtime.cancel(run_id)` signals the live in-process `CancellationToken` for
that run (if one is executing on this worker) in addition to persisting the
cancel request, so a running model/tool loop actually stops rather than only
being marked cancelled after the fact.

### Swarm

`linktools.ai.tasks.swarm` declares multi-agent swarms: a `SwarmSpec` of
member agents + a coordinator + a strategy + limits (`max_total_cost`,
`max_total_tokens`).

## Domain invariants

Domain models (`AgentSpec`, `ToolRef`, `ModelPolicy`, ...) validate their
contract at construction — a caller that builds one directly cannot create an
invalid object. Mapping fields are deep-frozen.

Canonical JSON (`linktools.ai.json.canonical_json`) is used for every hash and
fingerprint path — `default=str` is forbidden (it silently coerces arbitrary
objects into unstable / colliding strings).

## Tests

```bash
# from the repo root
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
```
