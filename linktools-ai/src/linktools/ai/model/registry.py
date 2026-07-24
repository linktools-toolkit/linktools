#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model configuration and pydantic-ai runtime factory.

This module owns:

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
    # gateways use custom paths that any normalization would corrupt, so
    # appending a trailing /v1 is opt-in via append_v1_if_missing.
    base_url_mode: "Literal['literal', 'append_v1_if_missing']" = "literal"

    @property
    def token(self) -> "str | None":
        return self.auth_token or self.api_key


@dataclass(slots=True)
class ModelBundle:
    """A configured model plus the per-call execution limits derived from config.

    Exposes ``revision`` (a stable hash of the non-secret config identity) so a
    run prepared against this bundle can detect provider drift on resume: a
    changed model_type / endpoint / protocol between prepare and resume is a
    real revision change, while key rotation (api_key / auth_token) is NOT --
    secrets are deliberately excluded from the hash so rotating them does not
    invalidate every resumable run."""

    config: RuntimeModelConfig
    model: OpenAIChatModel
    settings: ModelSettings
    usage_limits: UsageLimits

    @property
    def revision(self) -> str:
        import hashlib

        from ..json import canonical_json

        cfg = self.config
        # Only the non-secret config-identity fields. api_key / auth_token / the
        # opaque raw dict are excluded so the revision is stable under key
        # rotation and never persists a secret-derived value.
        identity = {
            "model_type": cfg.model_type,
            "protocol": cfg.protocol,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "timeout_seconds": cfg.timeout_seconds,
            "base_url_mode": cfg.base_url_mode,
        }
        return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()

    @classmethod
    def from_config(
        cls, config: "RuntimeModelConfig", *, request_retries: int
    ) -> "ModelBundle":
        """Build a config-backed OpenAI model bundle. The framework ALWAYS sets
        the provider HTTP client's ``max_retries`` to ``request_retries`` --
        including 0, which disables the SDK's own transient-HTTP retry. The
        value is never inferred from the client default. ``request_retries`` is a
        REQUEST-layer retry (the provider client retrying a transient HTTP
        failure), not a registry-lookup retry."""
        if config.protocol != "openai":
            raise ModelClientUnavailable(
                f"{config.model_type}: unsupported protocol '{config.protocol}' (use 'openai')"
            )
        from openai import AsyncOpenAI

        # Always pass max_retries explicitly, including 0 (which the SDK treats
        # as "do not retry"). Never fall back to the client's own default.
        client = AsyncOpenAI(
            base_url=_resolve_base_url(config),
            api_key=config.token,
            max_retries=request_retries,
        )
        provider = OpenAIProvider(openai_client=client)
        # The gateway routes to various OpenAI-compatible models, including
        # reasoning/"thinking mode" models (e.g. deepseek-v4-flash) that reject
        # `tool_choice: "required"` with HTTP 400. Disabling this lets pydantic-ai
        # fall back to `tool_choice: "auto"` for structured output.
        profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
        model = OpenAIChatModel(config.model or "", provider=provider, profile=profile)
        raw = config.raw
        settings = ModelSettings(
            max_tokens=int(raw.get("max_output_tokens", 4096)),
            timeout=float(config.timeout_seconds),
            parallel_tool_calls=True,
        )
        max_turns = int(raw.get("max_turns", 10))
        usage_limits = UsageLimits(request_limit=max(1, max_turns))
        return cls(config=config, model=model, settings=settings, usage_limits=usage_limits)

    @classmethod
    def from_instance(
        cls,
        model_type: str,
        model: "Model",
        *,
        request_retries: "int | None" = None,
        settings: "ModelSettings | None" = None,
        usage_limits: "UsageLimits | None" = None,
    ) -> "ModelBundle":
        """Wrap an already-constructed (prebuilt) Model. A prebuilt model owns its
        own HTTP client and retry behavior, so the framework cannot configure
        ``max_retries`` on it: ``request_retries`` MUST be None (the signal that
        the prebuilt model manages its own retries). A non-None value is a
        configuration error -- rejected explicitly, never silently ignored."""
        from ..errors import ModelRetryConfigurationError

        if request_retries is not None:
            raise ModelRetryConfigurationError(
                f"prebuilt model {model_type!r} cannot be configured with "
                f"request_retries={request_retries!r}; pass request_retries=None "
                f"so the prebuilt model manages its own retry behavior"
            )
        return cls(
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
            or ModelSettings(max_tokens=4096, timeout=300.0, parallel_tool_calls=True),
            usage_limits=usage_limits or UsageLimits(request_limit=10),
        )


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
            # Config-backed: registered with request_retries=0 (the framework
            # default); the resolver applies the policy's value at resolve time.
            bundle = ModelBundle.from_config(config, request_retries=0)
        else:
            # Prebuilt: no framework retry configuration (None).
            bundle = ModelBundle.from_instance(
                model_type,
                model,
                settings=settings,
                usage_limits=usage_limits,
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
    appending ``/v1`` is opt-in via ``base_url_mode``. Never strips
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
