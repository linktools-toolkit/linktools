#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentFeatureRegistry: the runtime kind -> AgentFeatureProvider store.

This is the single runtime registry the assembly domain keeps. What counts as a
valid feature kind is entirely determined by which providers are registered
here -- there is no separate hardcoded allowlist to keep in sync with the
actual provider set.

Registration lives here; per-spec *resolution* (turning an AgentSpec's tool
refs into one merged bundle, against this registry) is owned by the internal
:class:`~linktools.ai.agent.assembly.assembler.AgentAssembler`.
"""

from typing import Mapping

from ...errors import AgentAssemblyError, AgentFeatureConflictError
from .provider import AgentFeatureProvider, provider_kinds


class AgentFeatureRegistry:
    """Holds the kind -> AgentFeatureProvider mapping the resolver dispatches over.

    Constructed empty or seeded with a mapping; providers are added via
    :meth:`register` (strict -- fails if a kind is already taken) or
    :meth:`replace` (intentional override). A provider declaring multiple
    ``supported_kinds`` (e.g. ExtensionProvider) is registered under all of them
    from one call.
    """

    def __init__(self, providers: "Mapping[str, AgentFeatureProvider] | None" = None) -> None:
        self._providers: "dict[str, AgentFeatureProvider]" = dict(providers or {})
        self._frozen = False

    def __len__(self) -> int:
        return len(self._providers)

    @property
    def providers(self) -> "Mapping[str, AgentFeatureProvider]":
        # A copy so callers cannot mutate the registry's internal map.
        return dict(self._providers)

    def get(self, kind: str) -> "AgentFeatureProvider | None":
        return self._providers.get(kind)

    def register(self, provider: AgentFeatureProvider) -> None:
        """Register a provider for every kind it supports. Raises
        AgentFeatureConflictError if ANY of its kinds is already registered --
        silently overwriting a wired provider is never the right default. Call
        :meth:`replace` to override intentionally."""
        self._assert_mutable()
        kinds = provider_kinds(provider)
        for k in kinds:
            if k in self._providers:
                raise AgentFeatureConflictError(
                    f"feature provider already registered for kind {k!r}; "
                    f"use replace() to intentionally override it"
                )
        for k in kinds:
            self._providers[k] = provider

    def replace(self, provider: AgentFeatureProvider) -> None:
        """Register a provider for every kind it supports, intentionally
        overriding any provider already registered for those kinds."""
        self._assert_mutable()
        for k in provider_kinds(provider):
            self._providers[k] = provider

    def freeze(self) -> None:
        self._frozen = True

    def _assert_mutable(self) -> None:
        if self._frozen:
            raise AgentAssemblyError("AgentFeatureRegistry is frozen")


__all__: "list[str]" = ["AgentFeatureRegistry"]
