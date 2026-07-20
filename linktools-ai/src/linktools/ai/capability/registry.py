#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityProviderRegistry: the runtime kind -> CapabilityProvider store.

This is the single runtime registry the capability domain keeps (plan §4.3:
"只保留 CapabilityProviderRegistry 作为运行时 Registry"). What counts as a
valid capability kind is entirely determined by which providers are registered
here -- there is no separate hardcoded allowlist to keep in sync with the
actual provider set.

Registration lives here; per-spec *resolution* (turning an AgentSpec's tool
refs into one merged bundle, against this registry) is owned by the internal
:class:`~linktools.ai.capability.assembler.CapabilityAssembler`.
"""

from typing import Mapping

from ..errors import CapabilityConflictError
from .provider import CapabilityProvider, provider_kinds


class CapabilityProviderRegistry:
    """Holds the kind -> CapabilityProvider mapping the resolver dispatches over.

    Constructed empty or seeded with a mapping; providers are added via
    :meth:`register` (strict -- fails if a kind is already taken) or
    :meth:`replace` (intentional override). A provider declaring multiple
    ``supported_kinds`` (e.g. ExtensionProvider) is registered under all of them
    from one call.
    """

    def __init__(self, providers: "Mapping[str, CapabilityProvider] | None" = None) -> None:
        self._providers: "dict[str, CapabilityProvider]" = dict(providers or {})

    def __len__(self) -> int:
        return len(self._providers)

    @property
    def providers(self) -> "Mapping[str, CapabilityProvider]":
        # A copy so callers cannot mutate the registry's internal map.
        return dict(self._providers)

    def get(self, kind: str) -> "CapabilityProvider | None":
        return self._providers.get(kind)

    def register(self, provider: CapabilityProvider) -> None:
        """Register a provider for every kind it supports. Raises
        CapabilityConflictError if ANY of its kinds is already registered --
        silently overwriting a wired provider is never the right default. Call
        :meth:`replace` to override intentionally."""
        kinds = provider_kinds(provider)
        for k in kinds:
            if k in self._providers:
                raise CapabilityConflictError(
                    f"capability provider already registered for kind {k!r}; "
                    f"use replace() to intentionally override it"
                )
        for k in kinds:
            self._providers[k] = provider

    def replace(self, provider: CapabilityProvider) -> None:
        """Register a provider for every kind it supports, intentionally
        overriding any provider already registered for those kinds."""
        for k in provider_kinds(provider):
            self._providers[k] = provider


__all__: "list[str]" = ["CapabilityProviderRegistry"]
