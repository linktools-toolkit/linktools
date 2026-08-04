#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve a :class:`ModelPolicy` to a :class:`ResolvedModel` carrying the real
pydantic-ai Model to inject into the Agent.

Fallback lives at the REQUEST layer, not the registry-lookup layer. The resolver
walks primary then fallbacks ONCE -- a registry lookup is an in-memory read, not
a network call, so retrying it was meaningless. Unregistered candidates are
skipped (diagnostic, not fatal). With
two or more registered candidates, their models are wrapped in pydantic-ai's
:class:`~pydantic_ai.models.fallback.FallbackModel`, which tries each model at
request time and advances on :class:`~pydantic_ai.exceptions.ModelHTTPError`; a
single candidate is used directly.

``request_retries`` configures the provider HTTP client's own retry of transient
HTTP failures (wired into ``AsyncOpenAI`` as ``max_retries`` when a config-backed
OpenAI model is built, ALWAYS explicitly including 0). It is NOT a registry-lookup
retry. ``None`` is the signal that a prebuilt model (registered directly via
``model=``) manages its own retry behavior and is reused as-is; an int
``request_retries`` on a prebuilt model is rejected (the framework cannot configure
a client it does not own). The resolved revision is a stable hash of the ordered
candidates' non-secret identity plus ``request_retries``: reordering the chain,
swapping an endpoint field, or changing the retry count are real revision changes;
rotating an api_key is not."""


import hashlib
from dataclasses import dataclass
from ..errors import ModelRoutingError, ModelRetryConfigurationError
from ..json import canonical_json_bytes
from .registry import ModelBundle, ModelClientUnavailable, model_registry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.usage import UsageLimits
    from .policy import ModelPolicy
    from .registry import ModelRegistry


@dataclass(frozen=True)
class ResolvedModelCandidate:
    pricing_id: str
    provider_name: str
    model_name: str
    model: "Model"


@dataclass(frozen=True)
class ResolvedModel:
    """The real model + its stable revision + per-call limits, ready to inject
    into a pydantic-ai Agent."""

    model: "Model"
    candidates: "tuple[ResolvedModelCandidate, ...]"
    revision: str
    usage_limits: "UsageLimits"
    # Agent-layer structured-output retry count, sourced from the PRIMARY
    # candidate's bundle (fallbacks are not consulted, mirroring how
    # usage_limits is taken from bundles[0] only). Default 1 so direct
    # constructions that omit it (e.g. test stubs) keep pydantic-ai's default.
    output_retries: int = 1


class ModelResolver:
    """Resolve a :class:`ModelPolicy` to a :class:`ResolvedModel` by walking the
    candidate chain once and, for multiple registered candidates, wrapping them
    in a pydantic-ai ``FallbackModel`` (request-layer fallback)."""

    def __init__(self, *, registry: "ModelRegistry" = model_registry) -> None:
        self._registry = registry

    def resolve(self, policy: "ModelPolicy") -> ResolvedModel:
        bundles: "list[ModelBundle]" = []
        for model_type in (policy.primary, *policy.fallbacks):
            try:
                bundles.append(self._registry.get(model_type))
            except ModelClientUnavailable:
                # Unregistered candidate: skip (diagnostic, not fatal). A later
                # candidate may still satisfy the policy.
                continue
        if not bundles:
            raise ModelRoutingError(
                f"no registered model among primary={policy.primary!r} "
                f"fallbacks={policy.fallbacks!r}"
            )
        models = [_candidate_model(b, policy.request_retries) for b in bundles]
        candidates = tuple(
            _candidate_identity(bundle, model)
            for bundle, model in zip(bundles, models)
        )
        revision = _resolved_revision(bundles, policy.request_retries)
        usage_limits = bundles[0].usage_limits
        output_retries = bundles[0].output_retries
        if len(models) == 1:
            return ResolvedModel(
                models[0], candidates, revision, usage_limits, output_retries
            )
        from pydantic_ai.models.fallback import FallbackModel

        return ResolvedModel(
            FallbackModel(*models), candidates, revision, usage_limits, output_retries
        )


def _candidate_identity(
    bundle: ModelBundle, model: "Model"
) -> ResolvedModelCandidate:
    pricing_id = bundle.config.model or bundle.config.model_type
    provider_name = getattr(model, "provider_name", None)
    if not isinstance(provider_name, str):
        provider = getattr(model, "provider", None)
        provider_name = getattr(provider, "name", None)
    model_name = bundle.config.model or pricing_id
    if bundle.config.protocol == "prebuilt":
        model_identity = getattr(model, "model_name", None)
        if isinstance(model_identity, str) and model_identity:
            model_name = model_identity
    if not isinstance(provider_name, str) and ":" in pricing_id:
        provider_name, model_name = pricing_id.split(":", maxsplit=1)
    if not isinstance(provider_name, str):
        provider_name = ""
    return ResolvedModelCandidate(
        pricing_id=pricing_id,
        provider_name=provider_name,
        model_name=model_name,
        model=model,
    )


def resolve_pricing_id(
    *,
    provider_name: "str | None",
    response_model_name: "str | None",
    candidates: "tuple[ResolvedModelCandidate, ...]",
) -> "str | None":
    if response_model_name is None:
        return None
    matches = [
        candidate
        for candidate in candidates
        if candidate.model_name == response_model_name
        and (
            provider_name is None or candidate.provider_name == provider_name
        )
    ]
    return matches[0].pricing_id if len(matches) == 1 else None


def effective_request_retries(
    *,
    bundle: ModelBundle,
    requested: "int | None",
) -> "int | None":
    """The retry count that actually governs one candidate, normalized from the
    policy's raw ``requested`` value. This is the SINGLE source of truth shared by
    model selection (:func:`_candidate_model`) and revision computation
    (:func:`_resolved_revision`) so the two can never diverge -- a revision that
    disagrees with the executed retry count would let a resume mis-detect drift.

    Config-backed (``openai`` protocol) model: the framework owns the provider
    HTTP client, so ``None`` (the prebuilt signal) normalizes to ``0`` (the
    config-file default) and a non-negative ``int`` is taken as-is.

    Prebuilt model: it owns its own HTTP client, so ONLY ``None`` is accepted; an
    ``int`` asks the framework to configure a client it does not own and is
    rejected with :class:`ModelRetryConfigurationError`."""
    if bundle.config.protocol == "openai":
        return 0 if requested is None else requested
    if requested is not None:
        raise ModelRetryConfigurationError(
            f"prebuilt model {bundle.config.model_type!r} cannot be configured with "
            f"request_retries={requested!r}; pass request_retries=None so the "
            f"prebuilt model manages its own retry behavior"
        )
    return None


def _candidate_model(bundle: ModelBundle, request_retries: "int | None") -> "Model":
    """Pick the model for one candidate, applying the policy's retry semantics
    via :func:`effective_request_retries` (the shared normalization).

    Config-backed OpenAI model: the framework ALWAYS owns ``max_retries``. A
    registration builds with ``max_retries=0``; when the normalized count is
    positive the model is rebuilt so the provider client carries that count
    (0 is already explicit at registration, so the registered model is reused).

    Prebuilt model: it owns its own HTTP client, so the normalized value is
    ``None`` (reuse as-is); a non-None request is rejected inside
    :func:`effective_request_retries` before this branch is reached."""
    retries = effective_request_retries(bundle=bundle, requested=request_retries)
    if bundle.config.protocol == "openai":
        if retries > 0:
            return ModelBundle.from_config(
                bundle.config, request_retries=retries
            ).model
        return bundle.model
    return bundle.model


def _resolved_revision(
    bundles: "list[ModelBundle]", request_retries: "int | None"
) -> str:
    """Revision of the resolved model: a stable hash of the ordered candidates'
    own non-secret revisions plus each candidate's NORMALIZED retry count (via
    :func:`effective_request_retries`). Candidate order is significant, so
    reordering the fallback chain changes the revision; each candidate's own
    revision already excludes secrets, so key rotation does not. Normalizing per
    candidate means a config-backed model resolves ``None`` and ``0`` to the SAME
    revision -- the two are execution-equivalent and must be revision-equivalent
    too."""
    identity = [
        {
            "revision": b.revision,
            "request_retries": effective_request_retries(
                bundle=b, requested=request_retries
            ),
        }
        for b in bundles
    ]
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


__all__: "list[str]" = [
    "ResolvedModelCandidate",
    "ResolvedModel",
    "ModelResolver",
    "ModelRoutingError",
    "ModelRetryConfigurationError",
    "effective_request_retries",
    "resolve_pricing_id",
]
