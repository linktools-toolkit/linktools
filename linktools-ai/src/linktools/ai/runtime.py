#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface (spec section 5; deviation #3).
Runtime.build() assembles Storage + AgentCompiler + AgentRunner + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentRunner (AgentSpec) or
SwarmRunner (SwarmSpec)."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .agent_runtime.compiler import AgentCompiler
from .agent_runtime.runner import AgentRunner
from .agent_runtime.spec import AgentSpec
from .errors import SessionError, SwarmError
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .run.context import RunContext
from .run.models import RunInput, RunnableType
from .session.models import SessionRecord, SessionStatus
from .storage.facade import Storage
from .swarm_runtime.runner import SwarmRunner
from .swarm_runtime.spec import SwarmSpec


class Runtime:
    def __init__(self, *, storage: Storage, compiler: AgentCompiler,
                 runner: AgentRunner, swarm_runner: SwarmRunner,
                 model_router: ModelRouter) -> None:
        self.storage = storage
        self.compiler = compiler
        self.runner = runner
        self.swarm_runner = swarm_runner
        self.model_router = model_router

    @classmethod
    def build(cls, *, storage: Storage,
              model_router: "ModelRouter | None" = None,
              middleware_pipeline: "MiddlewarePipeline | None" = None,
              workspace_root: "str | Path | None" = None) -> "Runtime":
        router = model_router or ModelRouter()
        compiler = AgentCompiler(model_router=router, middleware_pipeline=middleware_pipeline)
        runner = AgentRunner(
            run_store=storage.runs, session_store=storage.sessions,
            event_store=storage.events, checkpoint_store=storage.checkpoints,
            middleware_pipeline=middleware_pipeline,
        )
        swarm_runner = SwarmRunner(
            swarm_store=storage.swarms,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            checkpoint_store=storage.checkpoints,
            compiler=compiler,
        )
        return cls(
            storage=storage, compiler=compiler, runner=runner,
            swarm_runner=swarm_runner, model_router=router,
        )

    async def run(self, spec: "AgentSpec | SwarmSpec", prompt: str, *,
                  session_id: "str | None" = None,
                  run_id: "str | None" = None,
                  user_id: "str | None" = None,
                  tenant_id: "str | None" = None,
                  agents: "Mapping[str, AgentSpec] | None" = None):
        resolved_session_id = session_id or str(uuid.uuid4())
        if session_id is not None:
            existing = await self.storage.sessions.get(session_id)
            if existing is None:
                raise SessionError(f"session not found: {session_id}")
        else:
            now = datetime.now(timezone.utc)
            await self.storage.sessions.create(SessionRecord(
                id=resolved_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
                created_at=now, updated_at=now))

        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError(
                    "agents mapping is required to run a SwarmSpec"
                )
            context = RunContext(
                run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
                session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.SWARM,
                user_id=user_id, tenant_id=tenant_id, workspace=None)
            return await self.swarm_runner.run(
                spec, RunInput(prompt=prompt), context, agents=agents,
            )

        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
            session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT,
            user_id=user_id, tenant_id=tenant_id, workspace=None)
        return await self.runner.run(compiled, RunInput(prompt=prompt), context)
