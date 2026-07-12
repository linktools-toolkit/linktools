#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model configuration and pydantic-ai runtime factory.

The hand-rolled OpenAI-compatible ReAct loop that previously lived here has been
replaced by pydantic-ai (see `base.py`). This module now owns:

- the shared error types raised/caught across the pipeline
  (`ModelClientUnavailable`, `ModelOutputError`, `ModelTurnLimitExceeded`);
- the `RuntimeModelConfig` dataclass, resolved by callers however they see fit
  (file, env vars, secrets manager, hardcoded for tests) and handed to
  `_bundle_from_config`;
- `_bundle_from_config`/`ModelBundle`, the pydantic-ai model factory built on top of
  `RuntimeModelConfig`; `ModelRegistry` also accepts a pre-built `Model` directly via
  `register(model_type, model=...)`.

`build_mcp_toolset`, mapping `MCPServerSpec` onto pydantic-ai `MCPToolset`s, now
lives in `..mcp.client` alongside the rest of the MCP wiring.

Session history persistence (context.json / per-call prompt sidecars) now lives in
`session/history.py`.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from linktools.core import environ

logger = environ.get_logger("ai.core.model.runtime")


class ModelClientUnavailable(RuntimeError):
    def __init__(self, message: str, diagnostics: "dict[str, Any] | None" = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelOutputError(RuntimeError):
    """Model response could not be parsed/validated as the expected structured output."""

    def __init__(self, message: str, diagnostics: "dict[str, Any] | None" = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelTurnLimitExceeded(RuntimeError):
    """The agent exhausted its per-call turn/request budget (UsageLimits.request_limit)
    without producing a result, typically from looping on tool calls."""

    def __init__(self, message: str, diagnostics: "dict[str, Any] | None" = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(slots=True)
class RuntimeModelConfig:
    model_type: str
    protocol: str
    model: "str | None"
    base_url: "str | None"
    api_key: "str | None"
    auth_token: "str | None"
    timeout_seconds: int
    raw: "dict[str, Any]"
    # base_url is passed through literally by default. Some OpenAI-compatible
    # gateways use custom paths that any normalization would corrupt, so the
    # prior "ensure trailing /v1" behavior is opt-in via append_v1_if_missing.
    base_url_mode: "Literal['literal', 'append_v1_if_missing']" = "literal"

    @property
    def token(self) -> "str | None":
        return self.auth_token or self.api_key


@dataclass(slots=True)
class ModelBundle:
    """A configured model plus the per-call execution limits derived from config."""

    config: RuntimeModelConfig
    model: OpenAIChatModel
    settings: ModelSettings
    usage_limits: UsageLimits


class ModelRegistry:
    """Process-wide model_type -> ModelBundle registry. Callers register every
    model_type an agent might request at startup — either a `RuntimeModelConfig`
    (resolved from file, env vars, secrets manager, or hardcoded for tests) or an
    already-constructed `Model` — and agents look up the resulting `ModelBundle` by
    model_type instead of resolving/building it on demand per call."""

    def __init__(self) -> None:
        self._bundles: "dict[str, ModelBundle]" = {}

    def register(
        self,
        model_type: str,
        *,
        config: "RuntimeModelConfig | None" = None,
        model: "Model | None" = None,
        settings: "ModelSettings | None" = None,
        usage_limits: "UsageLimits | None" = None,
    ) -> None:
        if (config is None) == (model is None):
            raise ValueError("register() requires exactly one of `config` or `model`")
        if config is not None:
            bundle = _bundle_from_config(config)
        else:
            bundle = ModelBundle(
                config=RuntimeModelConfig(
                    model_type=model_type,
                    protocol="prebuilt",
                    model=None,
                    base_url=None,
                    api_key=None,
                    auth_token=None,
                    timeout_seconds=300,
                    raw={},
                ),
                model=model,
                settings=settings
                or ModelSettings(
                    max_tokens=4096, timeout=300.0, parallel_tool_calls=True
                ),
                usage_limits=usage_limits or UsageLimits(request_limit=10),
            )
        self._bundles[model_type] = bundle

    def get(self, model_type: str) -> "ModelBundle":
        try:
            return self._bundles[model_type]
        except KeyError:
            raise ModelClientUnavailable(
                f"no model registered for model_type '{model_type}'"
            ) from None


model_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# pydantic-ai model factory
# ---------------------------------------------------------------------------


def _resolve_base_url(config: RuntimeModelConfig) -> str:
    """Resolve the OpenAI provider base_url. Literal pass-through by default;
    the prior auto-append ``/v1`` is opt-in via ``base_url_mode``. Never strips
    a user-supplied path or suffix -- some gateways rely on custom paths."""
    if not config.base_url:
        raise ModelClientUnavailable(
            f"{config.model_type}: openai protocol requires base_url"
        )

    mode = config.base_url_mode
    if mode == "literal":
        return config.base_url
    if mode == "append_v1_if_missing":
        base = config.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"
    raise ModelClientUnavailable(f"{config.model_type}: invalid base_url_mode {mode!r}")


def _bundle_from_config(config: RuntimeModelConfig) -> ModelBundle:
    """Build an `OpenAIChatModel` (+ settings/limits) for an already-resolved
    `RuntimeModelConfig`. Callers resolve configuration however they want (file,
    env vars, secrets manager, hardcoded for tests) and hand in the config directly
    — this function only does the pydantic-ai model/provider/settings construction."""
    if config.protocol != "openai":
        raise ModelClientUnavailable(
            f"{config.model_type}: unsupported protocol '{config.protocol}' (use 'openai')"
        )
    provider = OpenAIProvider(base_url=_resolve_base_url(config), api_key=config.token)
    # The gateway routes to various OpenAI-compatible models, including reasoning/
    # "thinking mode" models (e.g. deepseek-v4-flash) that reject `tool_choice:
    # "required"` with HTTP 400. Disabling this lets pydantic-ai fall back to
    # `tool_choice: "auto"` for structured output, which all backends accept.
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    model = OpenAIChatModel(config.model or "", provider=provider, profile=profile)

    raw = config.raw
    settings = ModelSettings(
        max_tokens=int(raw.get("max_output_tokens", 4096)),
        timeout=float(config.timeout_seconds),
        parallel_tool_calls=True,
    )
    # max_turns historically bounded the number of model requests per call; map it
    # onto pydantic-ai's request limit (one request per turn).
    max_turns = int(raw.get("max_turns", 10))
    usage_limits = UsageLimits(request_limit=max(1, max_turns))
    return ModelBundle(
        config=config, model=model, settings=settings, usage_limits=usage_limits
    )
