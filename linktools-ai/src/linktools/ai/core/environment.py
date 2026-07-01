#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The `AgentEnvironment` Protocol: the minimal contract the Agent runtime needs
from its host environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..support.hooks import HookRegistry


@runtime_checkable
class AgentEnvironment(Protocol):
    """Minimal contract the Agent runtime needs from its host environment.

    Concrete environments (e.g. sec-smartops-svc's `EngineEnvironment`) satisfy this
    structurally — no inheritance required.

    Member list is derived from actual `self.environ.*` usage in `engine/agent/agent.py`
    and `engine/agent/model_runtime.py` (source repo), not guessed:
    - `hooks`: read directly (`self.environ.hooks`) to fire lifecycle hook events.
    - `get_logger`: used to obtain a module logger (`environ.get_logger(name)`).
    - `env` / `config_root`: read transitively — `agent.py` calls
      `build_model(self.environ, model_type)`, which calls
      `load_runtime_model_config(env, model_type)`, which reads
      `env.config_root / f"config.{env.env}.yaml"`.

    Extended during Task 7 migration of `engine/agent/session.py` (re-verified against the
    full file, not just a grep) — `FileSession.create`/`DbSession.create` dereference two
    more members not caught by Task 5's grep from outside the file:
    - `workspace_root`: `environ.workspace_root`, the root passed to `RunContext`.
    - `trace_root(trace_id)`: `environ.trace_root(spec.trace_id)`, used to derive the
      per-trace runtime/session directories.
    """

    hooks: "HookRegistry | None"

    env: str

    config_root: Path

    workspace_root: Path

    def get_logger(self, name: str) -> logging.Logger: ...

    def trace_root(self, trace_id: str) -> Path: ...
