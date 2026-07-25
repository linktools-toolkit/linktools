#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunDispatcher: the narrow interface Job/Swarm/Subagent execution depends on
to run a compiled agent, instead of importing AgentEngine or Runtime-builder
internals directly.

The dispatcher is the SOLE authority for allocating a child run's identity and
for owning its full lifecycle (session create, RunContext build, snapshot
prepare, atomic start, claim/heartbeat/fencing, execute, terminal commit). A
caller (swarm strategy, subagent executor) describes WHAT to run and HOW its
session should be allocated; it must not mint child run ids, build child
RunContexts, create SessionRecords, or write RunDefinitions itself.

Two-step contract:
  1. ``open_child`` -- pure id + lineage allocation (no store writes). Returns
     a ``ChildRunHandle`` carrying the freshly-minted child run id, the derived
     session id, and the root/parent lineage. Allocating separately (rather
     than inside ``dispatch``) lets a caller record the child run id on its own
     domain state BEFORE execution starts -- e.g. the swarm strategy writes
     ``task.active_run_id`` so ``SwarmEngine.cancel()``/``recover()`` can locate
     the in-flight/stale child run. A crash between ``open_child`` and
     ``dispatch`` leaves no orphan RunRecord (nothing was persisted); recovery
     already tolerates "active_run_id set but Run missing".
  2. ``dispatch`` -- create the child session + RunRecord, prepare the
     resumable snapshot, then drive the full start/fence/execute/commit
     lifecycle and return the ``RunResult`` (raising ``RunPaused`` on a pause,
     propagating on failure).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, runtime_checkable

from .context import RunContext
from .models import RunInput, RunResult

if TYPE_CHECKING:
    from ..agent.models import CompiledAgent


@dataclass(frozen=True, slots=True)
class ChildSessionPolicy:
    """How the dispatcher allocates a child run's session. The dispatcher
    generates the session id and creates the SessionRecord; the caller only
    describes the policy.

    ``kind``:

    - ``"scratch"`` -- a fresh per-child session (the default). Workers must
      not touch the shared/parent session, so each child gets its own.
    - ``"shared"`` -- reuse the parent's session id (the dispatcher does not
      create a new SessionRecord).

    ``parent_session_id``: for a scratch session, the new SessionRecord is
    linked under this parent (``None`` = an orphan scratch session, e.g. a
    swarm worker).

    ``session_id_format``: an optional template for the scratch session id.
    ``{child_run_id}`` is filled by the dispatcher with the freshly-minted id;
    any other ``{field}`` placeholders are filled from the caller's
    ``metadata``. When ``None`` the dispatcher mints a plain uuid. (Swarm
    workers pass ``"swarm:{swarm_run_id}:{task_id}:{child_run_id}"`` so their
    scratch sessions keep a debuggable, stable naming.)
    """

    kind: Literal["scratch", "shared"] = "scratch"
    parent_session_id: "str | None" = None
    session_id_format: "str | None" = None


@dataclass(frozen=True, slots=True)
class ChildRunHandle:
    """The id + lineage ``open_child`` allocates for a child run. Pure
    allocation -- no store writes have occurred. ``dispatch`` consumes it to
    build the child RunContext and create the SessionRecord."""

    run_id: str
    session_id: str
    root_run_id: str
    parent_run_id: "str | None"
    parent_session_id: "str | None"
    user_id: "str | None"
    tenant_id: "str | None"
    # The workspace the child run executes under -- inherited from the parent
    # so a subagent/worker stays within the parent's workspace scope (sandbox,
    # workspace-scoped tools, permissions).
    workspace: "Any | None" = None
    # True when ``session_id`` is a freshly-allocated scratch session the
    # dispatcher must create (kind == "scratch"); False when it reuses the
    # parent's existing session (kind == "shared") and no creation is needed.
    session_needs_create: bool = True


@dataclass(frozen=True, slots=True)
class RunDispatchRequest:
    """Run a compiled agent under a child run whose identity was allocated by
    a prior ``open_child`` call. ``handle`` carries the child run id / session
    id / lineage; ``compiled_agent`` + ``input`` are what to run; ``metadata``
    is opaque caller context (e.g. swarm_run_id / task_id, the runnable id)."""

    compiled_agent: "CompiledAgent"
    input: RunInput
    handle: ChildRunHandle
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class RunDispatcher(Protocol):
    async def open_child(
        self,
        parent_context: RunContext,
        session_policy: ChildSessionPolicy,
        metadata: "Mapping[str, Any]",
    ) -> ChildRunHandle:
        """Allocate a child run's id + session id + lineage (pure, no store
        writes). See module docstring."""
        ...

    async def dispatch(self, request: RunDispatchRequest) -> RunResult:
        """Create the child session + RunRecord, prepare the snapshot, drive
        the full start/fence/execute/commit lifecycle, return the RunResult."""
        ...
