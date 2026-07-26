#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture invariants for the P1-P6 closure work.

These are HARD PASS rules (no xfail) pinning the refactor's structural
guarantees so a future commit cannot silently reintroduce a deleted
ambient-state field, a deleted resolver, or a path-based filesystem op the
SecureDirectory rewrite replaced."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "linktools-ai" / "src"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _source_files(*subpaths: str) -> "list[Path]":
    root = _SRC.joinpath(*subpaths)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.py"))


# --- SqlAlchemyObjectBackend invariants (P1 + P3) ---------------------------


SQL_OBJECT_BACKEND = _SRC / "linktools" / "ai" / "storage" / "backends" / "sqlalchemy" / "object.py"


def test_sql_object_backend_has_no_tx_session_attribute():
    """The reusable PARENT backend never carries active-session state. The old
    ``self._tx_session`` was the cross-coroutine bleed bug P1 closed."""
    source = _read(SQL_OBJECT_BACKEND)
    assert "_tx_session" not in source, (
        "SqlAlchemyObjectBackend reintroduced self._tx_session -- the "
        "ambient transaction state P1 deleted"
    )


def test_sql_object_backend_has_no_tx_revision_attribute():
    """The reusable PARENT backend never carries staged-revision state. The
    old ``self._tx_revision`` cached a revision across calls on the parent."""
    source = _read(SQL_OBJECT_BACKEND)
    assert "_tx_revision = None" not in source.replace(
        "self._tx_revision: ", "self._tx_revision: "
    ), (
        "SqlAlchemyObjectBackend reintroduced parent-level self._tx_revision "
        "default -- only the transaction child may cache a revision"
    )
    # The CHILD backend (_SqlAlchemyTransactionBackend) legitimately caches
    # _tx_revision in its own __init__; assert the parent's __init__ does not.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SqlAlchemyObjectBackend":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "__init__":
                    init_src = ast.unparse(item)
                    assert "_tx_revision" not in init_src, (
                        "SqlAlchemyObjectBackend.__init__ writes _tx_revision -- "
                        "only the transaction child should cache a revision"
                    )


def test_sql_object_backend_does_not_call_resolve_dialect_strategy():
    """P3: core never resolves a dialect from a session_factory. The backend
    receives the dialect explicitly."""
    source = _read(SQL_OBJECT_BACKEND)
    assert "resolve_dialect_strategy" not in source, (
        "SqlAlchemyObjectBackend calls resolve_dialect_strategy -- core must "
        "receive the dialect explicitly (P3 Protocol-first)"
    )


def test_sql_object_backend_does_not_branch_on_engine_dialect_name():
    """P3: the backend never reads engine.dialect.name to switch behavior."""
    source = _read(SQL_OBJECT_BACKEND)
    assert "engine.dialect.name" not in source, (
        "SqlAlchemyObjectBackend branches on engine.dialect.name -- core must "
        "stay dialect-neutral (P3 Protocol-first)"
    )


def test_dialects_package_ships_no_env_specific_modules():
    """P3: core ships no MySQL / PostgreSQL dialect modules. A downstream
    wanting one of those vendors ships its own."""
    dialects_dir = _SRC / "linktools" / "ai" / "storage" / "sqlalchemy" / "dialects"
    py_files = [p.name for p in dialects_dir.glob("*.py") if p.name != "__init__.py"]
    forbidden = {"mysql.py", "postgresql.py", "_hash_migration.py"}
    leaked = forbidden & set(py_files)
    assert not leaked, (
        f"storage/sqlalchemy/dialects/ leaked environment-specific modules: "
        f"{sorted(leaked)} -- core must not ship vendor dialects"
    )


def test_dialects_init_exports_no_env_specific_classes():
    """The package's public re-exports name no vendor dialect class."""
    init = _read(_SRC / "linktools" / "ai" / "storage" / "sqlalchemy" / "dialects" / "__init__.py")
    for forbidden in ("MySqlDialect", "PostgreSqlDialect", "resolve_dialect_strategy"):
        assert forbidden not in init, (
            f"dialects/__init__.py references {forbidden} -- a deleted API"
        )


# --- FilesystemObjectBackend invariants (P2) --------------------------------


FS_OBJECT_BACKEND = _SRC / "linktools" / "ai" / "storage" / "backends" / "filesystem" / "object.py"


@pytest.mark.parametrize(
    "forbidden_call",
    [
        # Path-typed calls are forbidden -- the backend reaches disk only via
        # self._sd.* (SecureDirectory's component-based, dirfd-relative API).
        # ``self._sd.read_bytes(`` is allowed; a bare ``.read_bytes(`` /
        # ``.read_text(`` on a Path is what the rule catches.
        "path.read_text",
        "path.read_bytes",
        "Path.read_text",
        "Path.read_bytes",
        "Path(...).read_text",
        "Path(...).read_bytes",
        "path.glob",
        "path.iterdir",
        "path.unlink",
        "path.mkdir",
        "path.replace",
        "Path(...).glob",
        "Path(...).iterdir",
        "Path(...).unlink",
        "Path(...).mkdir",
        "Path(...).replace",
        "os.replace(",  # Path-based os.replace bypasses the dirfd chain
    ],
)
def test_filesystem_object_backend_does_not_use_path_based_ops(forbidden_call: str):
    """P2: every filesystem op is dirfd-relative via SecureDirectory. Path-based
    read/write/glob/iterdir/unlink/mkdir/replace are forbidden in the backend's
    production CODE (docstrings/comments are excluded so prose mentions of the
    forbidden APIs do not trip the rule). ``self._sd.*`` is the only allowed
    filesystem surface."""
    source = _read(FS_OBJECT_BACKEND)
    code_text = _strip_comments_and_docstrings(source)
    assert forbidden_call not in code_text, (
        f"FilesystemObjectBackend uses path-based op {forbidden_call!r}; "
        f"all filesystem access must go through SecureDirectory (P2 dirfd-only)"
    )


def _strip_comments_and_docstrings(source: str) -> str:
    """Drop #-comments, module/class/function docstrings, and standalone
    string-literal expressions so architecture assertions can match CODE
    without false positives from prose mentions of the forbidden APIs."""
    tree = ast.parse(source)
    # Build a set of node ids to skip (docstring Expr statements + their
    # string-literal value).
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                body.pop(0)
    # Re-render module body; re-joining loses formatting but preserves code.
    return "\n".join(ast.unparse(stmt) for stmt in tree.body)


def test_filesystem_object_backend_does_not_import_resolve_secure_path():
    """P2: resolve_secure_path was deleted as the security boundary. The dirfd
    walk in SecureDirectory is the structural replacement. Imports + call sites
    are checked (docstring mentions are NOT)."""
    source = _read(FS_OBJECT_BACKEND)
    code_text = _strip_comments_and_docstrings(source)
    assert "resolve_secure_path" not in code_text, (
        "FilesystemObjectBackend imports/calls resolve_secure_path -- the "
        "path-check-then-use security boundary was deleted in P2"
    )


def test_path_security_module_is_deleted():
    """The whole _path_security module was deleted (SymlinkPolicy.ALLOW_INTERNAL
    and resolve_secure_path are both gone)."""
    path_security = _SRC / "linktools" / "ai" / "storage" / "backends" / "filesystem" / "_path_security.py"
    assert not path_security.exists(), (
        "_path_security.py still exists -- P2 (path-check-then-use deletion) deleted it as the security "
        "boundary; SecureDirectory is the structural replacement"
    )


# --- ObjectStore capability invariants (P4) ---------------------------------


OBJECT_STORE = _SRC / "linktools" / "ai" / "storage" / "object" / "store.py"


def test_object_store_exposes_capabilities_and_transaction_scope():
    """P4: ObjectStore exposes the public capability properties the
    consistency gate reads (no isinstance against concrete backends)."""
    source = _read(OBJECT_STORE)
    assert "def supports_optimistic_concurrency" in source
    assert "def transaction_scope" in source
    assert "def capabilities" in source


def test_storage_features_has_from_components_classmethod():
    """P4: StorageFeatures.from_components derives the capability
    frozensets from each wired store's capabilities, rather than defaulting
    to _ALL_COMPONENTS."""
    features = _read(_SRC / "linktools" / "ai" / "runtime" / "persistence" / "features.py")
    assert "def from_components" in features, (
        "StorageFeatures.from_components is missing -- P4 requires deriving "
        "features from wired stores, not optimistically declaring _ALL_COMPONENTS"
    )


# --- Live event hub invariants (P6) ----------------------------------------


LIVE_EVENTS = _SRC / "linktools" / "ai" / "run" / "live_events.py"


def test_live_events_does_not_push_a_close_sentinel():
    """P6: close() must not push a sentinel into the queue. The new design
    signals close via asyncio.Event and races it explicitly in publish/events."""
    source = _read(LIVE_EVENTS)
    assert "_CLOSED = object()" not in source, (
        "RunLiveStreamHandle reintroduced the _CLOSED sentinel -- P6 replaced "
        "it with an asyncio.Event + asyncio.wait race"
    )
    # The new design closes via an Event + asyncio.wait race in publish/events.
    tree = ast.parse(source)
    class_text = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RunLiveStreamHandle":
            class_text = ast.unparse(node)
            break
    assert "asyncio.Event" in class_text, (
        "RunLiveStreamHandle does not use asyncio.Event for closure -- the "
        "P6 cancellation-safe design requires it"
    )
    assert "asyncio.wait" in class_text, (
        "RunLiveStreamHandle does not race close against put/get via "
        "asyncio.wait -- the P6 cancellation-safe design requires it"
    )


# --- P5 the FS-journal-consistency spec Filesystem Run journal contract ------------------------------


FS_COMMIT = _SRC / "linktools" / "ai" / "run" / "persistence" / "commit.py"
JOURNAL = _SRC / "linktools" / "ai" / "run" / "persistence" / "journal.py"
RUNTIME_FACADE = _SRC / "linktools" / "ai" / "runtime" / "facade.py"


def test_fs_commit_coordinator_does_not_scan_events_for_dedup():
    """the FS-journal-consistency spec: the Filesystem Run commit coordinator must not dedup critical
    events by scanning the first 10,000 events. The dedup lives inside the
    EventStore's append_once index now, not the application layer."""
    source = _read(FS_COMMIT)
    assert "_event_exists" not in source, (
        "FilesystemRunCommitCoordinator still has an _event_exists helper -- "
        "the the FS-journal-consistency spec contract moved critical-event dedup into the EventStore's "
        "append_once index"
    )
    assert "limit=10000" not in source, (
        "FilesystemRunCommitCoordinator still scans a 10k-event window -- "
        "the FS-journal-consistency spec forbids that scan-based idempotency"
    )


def test_fs_commit_coordinator_threads_request_hash_through_every_commit():
    """the FS-journal-consistency spec: every commit method computes a request_hash and passes it to
    find_completion / find_incomplete / begin, so a retried call with the same
    commit_id but a different payload is a RunCommitConflictError rather than
    a silent overwrite."""
    source = _read(FS_COMMIT)
    for required in (
        'self._codec.request_hash("pause"',
        'self._codec.request_hash("complete"',
        'self._codec.request_hash("start"',
        'self._codec.request_hash("resume"',
        'self._codec.request_hash("fail"',
        'self._codec.request_hash("request_cancel"',
        'self._codec.request_hash("acknowledge_cancel"',
        "find_completion(",
        "request_hash=request_hash",
    ):
        assert required in source, (
            f"FilesystemRunCommitCoordinator missing {required!r} -- the "
            "the FS-journal-consistency spec contract requires every commit method to compute + thread "
            "a request_hash for replay-vs-conflict detection"
        )


def test_journal_records_request_hash_and_completion():
    """the FS-journal-consistency spec: the journal TransactionRecord carries a request_hash field,
    and the journal exposes record_completion / find_completion so a retried
    call with the same (commit_id, request_hash) returns the original result."""
    source = _read(JOURNAL)
    assert "request_hash: str" in source, (
        "TransactionRecord does not carry a request_hash field -- the FS-journal-consistency spec "
        "requires the journal to save the request hash for conflict detection"
    )
    for required in ("def find_completion", "def record_completion"):
        assert required in source, (
            f"TransactionJournal missing {required!r} -- the FS-journal-consistency spec requires a "
            "stable completion log so a retried commit returns the original "
            "result instead of re-executing"
        )


def test_run_commit_coordinator_protocol_requires_recovery():
    """the FS-journal-consistency spec: the RunCommitCoordinator Protocol declares
    recover_incomplete_commits() so the build kernel can gate Runtime start
    on it. SQL provides a no-op; Filesystem provides real recovery."""
    source = _read(_SRC / "linktools" / "ai" / "run" / "commit.py")
    assert "def recover_incomplete_commits" in source, (
        "RunCommitCoordinator Protocol does not declare "
        "recover_incomplete_commits -- the FS-journal-consistency spec requires recovery to run before "
        "Runtime accepts any new request"
    )


def test_runtime_lazily_runs_recovery_before_lifecycle_entries():
    """the FS-journal-consistency spec: Runtime.run/run_stream/cancel/approve/reject/resume and
    __aenter__ each await _ensure_recovered() before forwarding, so a
    crash-left in-flight commit is reconciled before the Runtime accepts a
    new request -- even when the caller did not enter via ``async with``."""
    source = _read(RUNTIME_FACADE)
    assert "_ensure_recovered" in source, (
        "Runtime does not gate lifecycle entries on _ensure_recovered -- the FS-journal-consistency spec "
        "requires recovery to run before the Runtime accepts any new request"
    )
    # Every lifecycle entry point must call _ensure_recovered() before forwarding.
    for entry in (
        "async def run(",
        "async def run_stream(",
        "async def cancel(",
        "async def approve(",
        "async def reject(",
        "async def resume(",
        "async def __aenter__(",
    ):
        assert entry in source, f"Runtime missing entry {entry!r}"
    # Count the calls — should be one per entry.
    calls = source.count("await self._ensure_recovered()")
    assert calls >= 7, (
        f"Runtime only has {calls} _ensure_recovered() calls; expected ≥7 "
        "(run/run_stream/cancel/approve/reject/resume/__aenter__)"
    )


def test_event_store_protocol_requires_append_once():
    """the event-store idempotency spec: EventStore Protocol declares append_once(commit_id=...) so a
    commit-scoped idempotent append reserves (stream_id, commit_id,
    event_type) inside the store rather than via an application-layer scan."""
    source = _read(_SRC / "linktools" / "ai" / "events" / "store.py")
    assert "async def append_once(" in source, (
        "EventStore Protocol does not declare append_once -- the event-store idempotency spec requires "
        "the store to own commit-scoped idempotent append"
    )


def test_fs_event_store_append_once_uses_event_type_in_index_key():
    """The Filesystem EventStore's append_once index MUST key on
    (stream_id, commit_id, event_type) so a multi-event commit (pause emits
    ApprovalRequested + RunPaused) does not collapse the second event onto
    the first."""
    source = _read(_SRC / "linktools" / "ai" / "events" / "persistence" / "filesystem.py")
    assert "event_type" in source, (
        "FilesystemEventStore append_once index does not include event_type "
        "in the dedup key -- a multi-event commit would collapse its events"
    )
    assert "_commit_index_path" in source and "event_type" in source, (
        "FilesystemEventStore commit-index path helper must take event_type"
    )


# --- P6 the fenced-security-event spec + the state-event split spec Fenced security event + state-event split -----------


AGENT_ENGINE = _SRC / "linktools" / "ai" / "agent" / "engine.py"
RUNTIME_BUILDER = _SRC / "linktools" / "ai" / "runtime" / "builder.py"


def test_agent_engine_does_not_publish_state_events():
    """the state-event split spec: AgentEngine publishes ONLY process events (text/tool/
    model_progress); it must NOT publish the paused/completed/failed/
    cancelled state events -- those are the RunCoordinator's job, emitted
    only AFTER the durable commit succeeds.

    Asserts the absence of a "type": "paused" / "completed" / "failed" /
    "cancelled" literal in the engine source (a state-event publish would
    necessarily construct such a dict)."""
    source = _read(AGENT_ENGINE)
    for forbidden in (
        '"type": "paused"',
        '"type": "completed"',
        '"type": "failed"',
        '"type": "cancelled"',
    ):
        assert forbidden not in source, (
            f"AgentEngine publishes a state event ({forbidden!r}) -- the state-event split spec "
            "forbids that; the engine publishes ONLY process events, the "
            "Coordinator publishes state events after the durable commit"
        )


def test_agent_engine_assertion_error_catch_is_narrowed():
    """the AssertionError-narrowing spec: the blanket ``except AssertionError: pass`` must be gone.
    The engine may either narrow the catch to the streaming-not-supported
    message or (preferred) let AssertionError propagate entirely as the
    invariant error it is."""
    source = _read(AGENT_ENGINE)
    # The forbidden blanket form: ``except AssertionError:\n    pass``
    assert re.search(
        r"except AssertionError[^:]*:\s*\n\s*pass\b", source
    ) is None, (
        "AgentEngine still has a blanket `except AssertionError: pass` -- "
        "the AssertionError-narrowing spec requires either catching only the streaming-not-supported "
        "message or letting AssertionError propagate"
    )


def test_run_coordinator_takes_fenced_event_writer():
    """the fenced-security-event spec: RunCoordinator accepts a fenced_event_writer dep so it can
    construct a per-execution SecurityEventSink that verifies the claiming
    execution's fence BEFORE appending a security event."""
    coord = _read(_SRC / "linktools" / "ai" / "run" / "coordinator.py")
    assert "fenced_event_writer" in coord, (
        "RunCoordinator does not accept a fenced_event_writer dep -- the fenced-security-event spec "
        "requires per-execution SecurityEventSink construction over a "
        "FencedRunEventWriter"
    )
    assert "_FencedSecurityEventSink" in coord, (
        "RunCoordinator does not wire a fenced SecurityEventSink -- the "
        "per-execution sink must verify the fence before forwarding a "
        "security event to the EventStore"
    )


def test_filesystem_fenced_run_event_writer_exists():
    """the fenced-security-event spec: the Filesystem FencedRunEventWriter implementation must exist
    alongside the SQL one so both backends can verify fences before appending
    security events."""
    fs_writer = (
        _SRC
        / "linktools"
        / "ai"
        / "run"
        / "persistence"
        / "filesystem"
        / "event_writer.py"
    )
    assert fs_writer.exists(), (
        "Filesystem FencedRunEventWriter implementation missing -- the fenced-security-event spec "
        "requires both SQL and Filesystem impls"
    )
    source = _read(fs_writer)
    assert "class FilesystemFencedRunEventWriter" in source
    assert "RunFenceLostError" in source, (
        "FilesystemFencedRunEventWriter must raise RunFenceLostError on a "
        "stale/empty/missing fence (the security-sensitive action that "
        "triggered the append must NOT proceed)"
    )


def test_runtime_builder_wires_fenced_event_writer_into_coordinator():
    """the fenced-security-event spec: build_runtime (the composition root)
    dispatches the backend-appropriate FencedRunEventWriter from the Storage
    and threads it into RunCoordinator via RuntimeBuildConfig so production
    code always fences security events. The dispatch lives in the composition
    root, not the build kernel -- the build kernel assembles only and never
    branches on Storage type."""
    facade_src = _read(_SRC / "linktools" / "ai" / "runtime" / "facade.py")
    assert "_resolve_fenced_event_writer_for" in facade_src, (
        "build_runtime does not resolve a FencedRunEventWriter -- the "
        "fenced-security-event spec requires per-execution fenced security-"
        "event appending in production builds"
    )
    assert "SqlAlchemyFencedRunEventWriter" in facade_src
    assert "FilesystemFencedRunEventWriter" in facade_src
    # And the build kernel reads it from the config without dispatching.
    builder_src = _read(RUNTIME_BUILDER)
    assert "fenced_event_writer=config.fenced_event_writer" in builder_src, (
        "build_runtime_components should read fenced_event_writer from the "
        "config (composition-root-injected), not dispatch a backend itself"
    )


# --- P7 swarm-commit-boundary: dep-cleanup + child-run chain -------------


SWARM_ENGINE = _SRC / "linktools" / "ai" / "swarm" / "engine.py"
SWARM_STRATEGY = _SRC / "linktools" / "ai" / "swarm" / "strategy.py"
DISPATCHER = _SRC / "linktools" / "ai" / "runtime" / "dispatcher.py"


@pytest.mark.parametrize(
    "forbidden_import",
    [
        "from ..run.store import RunStore",
        "from ..session.store import SessionStore",
        "from ..events.store import EventStore",
        "from ..run.definition import RunDefinitionStore",
        "from ..run.controller import RunController",
        "from ..events.context import EventStreamContext, append_event",
        "from ..events.context import append_event",
        "from ..run.lifecycle import mark_completed, mark_failed",
        "from ..run.lifecycle import mark_completed",
        "from ..run.lifecycle import mark_failed",
    ],
)
def test_swarm_engine_does_not_import_run_lifecycle_stores(forbidden_import: str):
    """the swarm-engine dep-cleanup spec: SwarmEngine must not import any
    Run/Event/Session/Definition Store or the lifecycle helpers
    (append_event / mark_completed / mark_failed). The driving Run's
    lifecycle is the RunCoordinator's job."""
    source = _read(SWARM_ENGINE)
    assert forbidden_import not in source, (
        f"SwarmEngine imports {forbidden_import!r} -- the swarm-engine "
        "dep-cleanup spec forbids any Run/Event/Session/Definition Store "
        "or lifecycle helper import"
    )


def test_swarm_strategy_does_not_persist_child_run_records():
    """the child-run spec: SwarmStrategy submits RunDispatchRequest through
    the dispatcher; it does NOT create or persist child RunRecords directly."""
    source = _read(SWARM_STRATEGY)
    code = _strip_comments_and_docstrings(source)
    for forbidden in (
        "RunStore.create",
        "SessionStore.create",
        "runs.create(",
        "sessions.create(",
    ):
        assert forbidden not in code, (
            f"SwarmStrategy persists a child run directly via {forbidden!r} "
            "-- the child-run spec requires it to submit RunDispatchRequest "
            "through the dispatcher"
        )


def test_dispatcher_chain_pins_coordinator_not_agent_engine():
    """the child-run boundary spec: the LateBoundRunDispatcher must bind to a
    CoordinatorRunDispatcher (which delegates to RunCoordinator), never to
    an AgentEngine."""
    dispatcher_src = _read(DISPATCHER)
    assert "class CoordinatorRunDispatcher" in dispatcher_src, (
        "CoordinatorRunDispatcher missing -- the dispatcher chain requires it"
    )
    coord_class = _extract_class(dispatcher_src, "CoordinatorRunDispatcher")
    assert "_coordinator" in coord_class, (
        "CoordinatorRunDispatcher does not wrap a RunCoordinator -- the "
        "child-run boundary spec requires it"
    )
    builder_src = _read(RUNTIME_BUILDER)
    assert "CoordinatorRunDispatcher(run_coordinator)" in builder_src, (
        "build_runtime_components does not bind LateBoundRunDispatcher to a "
        "CoordinatorRunDispatcher -- the child-run boundary spec requires "
        "the chain LateBoundRunDispatcher -> CoordinatorRunDispatcher -> "
        "RunCoordinator, no rebinding to AgentEngine"
    )


def test_swarm_engine_accepts_commit_coordinator_dep():
    """the swarm-commit-boundary spec: SwarmEngine takes a
    swarm_commit_coordinator dep so its lifecycle entry write (create_run)
    can route through the coordinator for commit_id-keyed idempotency."""
    source = _read(SWARM_ENGINE)
    assert "swarm_commit_coordinator" in source, (
        "SwarmEngine does not accept swarm_commit_coordinator -- the "
        "swarm-commit-boundary spec requires the lifecycle entry write to "
        "route through the coordinator"
    )
    # swarm_commit_coordinator must be a REQUIRED dep, no default None.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SwarmEngine":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "__init__":
                    found_kw = False
                    for default in item.args.defaults:
                        # A kw-only default named None means a fallback -- not allowed.
                        if isinstance(default, ast.Constant) and default.value is None:
                            # We don't know which arg this is the default for; check
                            # the engine source directly for the forbidden shape.
                            pass
                    found_kw = True
    assert "swarm_commit_coordinator: \"SwarmCommitCoordinator\"," in source or (
        "swarm_commit_coordinator: SwarmCommitCoordinator," in source
    ), (
        "SwarmEngine.swarm_commit_coordinator must be typed as "
        "SwarmCommitCoordinator with NO default None fallback -- the spec's "
        "strict end-state forbids the direct-store fallback"
    )
    # The engine must NOT carry a `= None` default on swarm_commit_coordinator
    # (the transitional fallback that the spec end-state removes).
    assert "swarm_commit_coordinator: " in source
    forbidden = re.search(
        r"swarm_commit_coordinator:\s*[^=,\n]+\s*=\s*None", source
    )
    assert forbidden is None, (
        "SwarmEngine.swarm_commit_coordinator still carries a `= None` default "
        "-- the spec end-state makes it required; remove the fallback"
    )
    builder_src = _read(RUNTIME_BUILDER)
    assert "swarm_commit_coordinator=_resolve_swarm_commit_coordinator" in builder_src
    facade_src = _read(_SRC / "linktools" / "ai" / "runtime" / "facade.py")
    assert "_resolve_swarm_commit_coordinator_for" in facade_src, (
        "build_runtime does not resolve a SwarmCommitCoordinator -- the "
        "swarm-commit-boundary spec requires production builds to wire the "
        "backend-appropriate coordinator into SwarmEngine"
    )
    # And the build kernel reads from config without dispatching.
    assert "config.swarm_commit_coordinator" in builder_src, (
        "build_runtime_components should read swarm_commit_coordinator from "
        "the config (composition-root-injected), not dispatch a backend itself"
    )


def test_swarm_domain_keeps_swarmstore_for_mid_execution_ops():
    """the swarm-engine dep-cleanup spec forbids only RUN-domain stores
    (RunStore/SessionStore/EventStore/RunDefinitionStore) + the lifecycle
    helpers (append_event/mark_completed/mark_failed) from SwarmEngine +
    SwarmStrategy. SwarmStore itself is the SWARM-domain store and stays in
    use: claim_task / set_active_run / list_tasks are legitimate mid-execution
    swarm operations, NOT lifecycle commits the SwarmCommitCoordinator owns."""
    strategy_src = _read(SWARM_STRATEGY)
    # SwarmStore references stay (it's the swarm-domain store).
    assert "swarm_store" in strategy_src, (
        "SwarmStrategy no longer references swarm_store -- the dep-cleanup "
        "spec forbids only RUN-domain stores, not SwarmStore; mid-execution "
        "ops (claim_task, set_active_run, list_tasks) legitimately use it"
    )
    # No RUN-domain store references.
    for forbidden in (
        "run_store",
        "session_store",
        "event_store",
        "run_definitions",
        "append_event",
        "mark_completed",
        "mark_failed",
    ):
        assert forbidden not in strategy_src, (
            f"SwarmStrategy references {forbidden!r} -- the dep-cleanup spec "
            "forbids RUN-domain stores + lifecycle helpers in the swarm domain"
        )


def _extract_class(source: str, class_name: str) -> str:
    """Return the ast.unparse text of a class definition by name (or empty
    string if not found)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.unparse(node)
    return ""


# --- P8 architecture-rules: Run commit commands (no Any, no empty commit_id) ---


RUN_COMMIT = _SRC / "linktools" / "ai" / "run" / "commit.py"


@pytest.mark.parametrize(
    "command_class",
    [
        "StartRunCommand",
        "PauseRunCommand",
        "ResumeRunCommand",
        "CompleteRunCommand",
        "FailRunCommand",
        "RequestCancelRunCommand",
        "AcknowledgeCancelRunCommand",
    ],
)
def test_run_commit_command_has_no_bare_any_field(command_class: str):
    """the run-commit-commands spec rule: no field on a Run commit command
    dataclass may be typed as bare ``Any`` (untyped scalar). ``Mapping[str,
    Any]`` is permitted (typed container, permissive value type); a bare
    ``record: Any`` / ``result: Any`` / ``event_context: Any`` is forbidden --
    each must name the concrete type the coordinator consumes."""
    source = _read(RUN_COMMIT)
    class_body = _extract_class(source, command_class)
    assert class_body, f"{command_class} not found in run/commit.py"
    # Walk the class body's annotated assignments; flag any whose annotation
    # is the bare name Any (not inside a subscript / Container).
    tree = ast.parse(class_body)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ann = node.annotation
            # Bare `Any` Name node -> forbidden.
            if isinstance(ann, ast.Name) and ann.id == "Any":
                pytest.fail(
                    f"{command_class}.{node.target.id} is typed as bare Any -- "
                    "the run-commit-commands spec rule forbids untyped scalar "
                    "fields; name the concrete type"
                )


def test_run_commit_command_commit_id_is_typed_required_runcommitid():
    """the run-commit-commands spec rule: ``commit_id`` must be the typed
    RunCommitId value object with NO default. The empty-string default that
    defeated idempotent replay is structurally forbidden."""
    source = _read(RUN_COMMIT)
    # The typed value object exists.
    assert "class RunCommitId:" in source
    # No commit_id field carries a str type or an empty default.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Command"):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ) and item.target.id == "commit_id":
                    # Must be typed RunCommitId (Name) -- not str.
                    ann = item.annotation
                    bare = isinstance(ann, ast.Name) and ann.id == "str"
                    # Must NOT carry a default (an = node.value would be the
                    # empty-string bypass).
                    has_default = item.value is not None
                    assert not bare, (
                        f"{node.name}.commit_id is typed `str` -- the rule "
                        "requires the typed RunCommitId value object"
                    )
                    assert not has_default, (
                        f"{node.name}.commit_id carries a default -- the "
                        "empty-commit_id-bypass rule forbids defaults on "
                        "commit_id (it must be required)"
                    )


def test_run_commit_execution_fence_is_non_empty_value_object():
    """the run-commit-commands spec rule: a fence is the typed ExecutionFence
    value object whose token CANNOT be empty (the empty-token-bypass-fencing
    bug). The value object's __post_init__ rejects an empty token."""
    source = _read(RUN_COMMIT)
    assert "class ExecutionFence:" in source
    class_body = _extract_class(source, "ExecutionFence")
    # __post_init__ must raise on an empty token.
    assert "__post_init__" in class_body
    assert "if not self.token" in class_body, (
        "ExecutionFence.__post_init__ does not reject an empty token -- the "
        "empty-token-bypass-fencing rule requires it"
    )
