#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolve the stable AI config through the real ``ConfigField`` framework.

Resolution priority (highest first): explicit CLI value → environment variable →
project ``.linktools/config.yaml`` → JSON cache → interactive prompt → error.
Base URL / model are cached (asked once); the API key is never cached.

The spec names this layer "CachedPromptProvider"; the real framework has no such
class -- caching is ``PromptProvider(cached=True)`` writing through a
``PersistentSource``. This module is that concept realized as a resolver over a
``Config`` wired with the right source order."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from linktools.core import (
    Config,
    ConfigSchema,
    DefaultSource,
    EnvironmentSource,
    FileSource,
    PersistentSource,
    RuntimeOverrideSource,
)
from linktools.core._config_store import ConfigStore
from linktools.errors import ConfigError
from linktools.rich import is_no_input, set_no_input

from .errors import MissingConfigError
from .fields import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    REMOTE_RUNTIME_URL,
)


@dataclass(frozen=True, slots=True)
class ResolvedAiConfig:
    base_url: str
    model: str
    api_key: str
    runtime_url: "str | None"


def _project_config_dict(config_yaml_path: "Path | str | None") -> dict:
    """Flatten ``.linktools/config.yaml`` into the env-style field names.

    Honors a natural ``openai:`` / ``runtime:`` section and top-level
    ``OPENAI_*`` keys. Returns an empty dict when the file is absent (project
    config is optional in the resolution priority)."""
    if not config_yaml_path:
        return {}
    path = Path(config_yaml_path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    flat: dict = {}
    openai = data.get("openai") or {}
    if isinstance(openai, dict):
        flat["OPENAI_BASE_URL"] = openai.get("base_url")
        flat["OPENAI_MODEL"] = openai.get("model")
        flat["OPENAI_API_KEY"] = openai.get("api_key")
    runtime = data.get("runtime") or {}
    if isinstance(runtime, dict):
        flat["LINKTOOLS_AI_RUNTIME_URL"] = runtime.get("url")
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "LINKTOOLS_AI_RUNTIME_URL",
    ):
        if data.get(key) is not None:
            flat[key] = data[key]
    return {k: v for k, v in flat.items() if v is not None}


def resolve_ai_config(
    *,
    base_url: "str | None" = None,
    model: "str | None" = None,
    api_key: "str | None" = None,
    runtime_url: "str | None" = None,
    config_yaml_path: "Path | str | None" = None,
    cache_store: ConfigStore,
    interactive: bool = True,
) -> ResolvedAiConfig:
    """Resolve base_url/model/api_key/runtime_url per the layered priority.

    ``interactive=False`` (e.g. ``--json`` / CI) makes a missing required value
    raise :class:`MissingConfigError` instead of prompting. Explicit CLI values
    seed the highest-priority source and are never written back to the cache."""
    schema = ConfigSchema()
    for field in (OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY, REMOTE_RUNTIME_URL):
        schema.define(field)

    runtime = RuntimeOverrideSource()
    if base_url:
        runtime.set("OPENAI_BASE_URL", base_url)
    if model:
        runtime.set("OPENAI_MODEL", model)
    if api_key:
        runtime.set("OPENAI_API_KEY", api_key)
    if runtime_url:
        runtime.set("LINKTOOLS_AI_RUNTIME_URL", runtime_url)

    project = FileSource(_project_config_dict(config_yaml_path), name="project-config")
    config = Config(
        None,
        schema,
        sources=[
            runtime,  # explicit CLI flags
            EnvironmentSource((os.environ, "")),  # OPENAI_* / LINKTOOLS_AI_* env
            project,  # .linktools/config.yaml
            PersistentSource(cache_store, "ai"),  # ~/.linktools/config/ai.json
            DefaultSource(schema),  # field.default fallback
        ],
    )

    # Respect the framework's --yes flag (NoInputAction sets _no_input globally
    # at parse time). Snapshot it, force no-input only when explicitly
    # non-interactive, and restore afterwards so a long-lived process (e.g. the
    # TUI) is never left in no-input mode by this call.
    prior_no_input = is_no_input()
    if not interactive:
        set_no_input(True)
    try:
        resolved_base = config.require("OPENAI_BASE_URL")
        resolved_model = config.get("OPENAI_MODEL", default="") or ""
        resolved_key = config.require("OPENAI_API_KEY")
        resolved_runtime = config.get("LINKTOOLS_AI_RUNTIME_URL", default=None)
    except ConfigError as exc:
        raise MissingConfigError(str(exc)) from exc
    finally:
        set_no_input(prior_no_input)
    return ResolvedAiConfig(
        base_url=resolved_base,
        model=resolved_model,
        api_key=resolved_key,
        runtime_url=resolved_runtime,
    )
