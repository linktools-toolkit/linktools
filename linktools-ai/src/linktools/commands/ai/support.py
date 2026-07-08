#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for the `lt ai` command (model config, runtime, agent spec)."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import CommandError
from linktools.core import environ
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.execution.local import LocalExecutionBackend
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import RuntimeModelConfig, model_registry
from linktools.ai.model.router import ModelRouter
from linktools.ai.runtime import Runtime
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FileStorage

if TYPE_CHECKING:
    from argparse import Namespace

# Mirrors the legacy chat prompt. Kept here so every `lt ai` subcommand that
# builds an agent shares the same persona without re-importing the CLI module.
SYSTEM_PROMPT = (
    "You are a general-purpose local assistant running in a terminal. "
    "You can read/write files and run shell commands in the current working "
    "directory via your file and terminal tools. Be direct and concise."
)


def resolve_model_config(
    model: "str | None",
    base_url: "str | None",
    api_key: "str | None",
) -> RuntimeModelConfig:
    """Build a `RuntimeModelConfig` from CLI flags, falling back to env vars.

    Flags win over environment variables. Raises `CommandError` if `base_url`
    or `api_key` are unset after both sources are checked.
    """
    resolved_model = model or os.environ.get("OPENAI_MODEL") or ""
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not resolved_base_url:
        raise CommandError(
            "no base url provided: pass --base-url or set OPENAI_BASE_URL"
        )
    if not resolved_api_key:
        raise CommandError(
            "no api key provided: pass --api-key or set OPENAI_API_KEY"
        )

    return RuntimeModelConfig(
        model_type="standard",
        protocol="openai",
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        auth_token=None,
        timeout_seconds=300,
        raw={},
    )


def validate_session_id(session_id: str) -> str:
    """Reject session ids that could escape the sessions directory when joined
    into a filesystem path. Raises CommandError with a clear message; returns
    session_id unchanged if it's safe.
    """
    is_unsafe = (
        not session_id
        or "/" in session_id
        or "\\" in session_id
        or session_id in (".", "..")
    )
    if is_unsafe:
        raise CommandError(
            f'invalid --session value "{session_id}": must not contain path separators or ".."'
        )
    return session_id


def build_storage() -> FileStorage:
    """Build the shared file-backed storage rooted at the ai data directory."""
    return FileStorage(root=environ.get_data_path("ai"))


def build_runtime(args: "Namespace") -> Runtime:
    """Resolve model config, register it, and build a `Runtime` wired to the
    shared file storage and the agent working directory.

    `args` is expected to expose ``model``, ``base_url``, ``api_key`` and
    ``workdir`` attributes (any falsy values fall back to defaults/env).
    """
    config = resolve_model_config(args.model, args.base_url, args.api_key)
    model_registry.register(config.model_type, config=config)
    workdir = Path(args.workdir) if args.workdir else Path.cwd()
    storage = build_storage()
    return Runtime.build(
        storage=storage,
        model_router=ModelRouter(registry=model_registry),
        execution=LocalExecutionBackend(runtime_dir=workdir),
    )


def build_agent_spec(args: "Namespace") -> AgentSpec:
    """Build the default `AgentSpec` used by `lt ai chat` / `lt ai run`.

    Re-resolves the model config so this helper only depends on `args` (mirrors
    the cntr pattern where each subcommand receives a single args object).
    """
    config = resolve_model_config(args.model, args.base_url, args.api_key)
    return AgentSpec(
        id="ai", name="ai",
        model=ModelPolicy(primary=config.model_type),
        instructions=PromptSpec(instructions=SYSTEM_PROMPT),
        output_schema=str,
    )


async def ensure_session(storage: FileStorage, session_id: str) -> None:
    """Get-or-create a session record.

    `Runtime.run` / `Runtime.run_stream` require a pre-existing session when a
    `session_id` is supplied (they do not auto-create). This mirrors the
    `session_id=None` branch exactly by creating the `SessionRecord` up-front
    when the id is unseen.
    """
    if await storage.sessions.get(session_id) is None:
        now = datetime.now(timezone.utc)
        await storage.sessions.create(SessionRecord(
            id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
            version=1, created_at=now, updated_at=now,
        ))
