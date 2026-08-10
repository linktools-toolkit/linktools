#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable capability resolver registry."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ..errors import AIError, ErrorCode
from ._contract import CapabilityResolver, validate_fingerprint


class CapabilityResolverRegistry:
    def __init__(self, resolvers: Sequence[CapabilityResolver]) -> None:
        values = tuple(resolvers)
        providers: dict[str, CapabilityResolver] = {}
        for resolver in values:
            provider = resolver.provider.strip()
            if not provider:
                raise ValueError("capability resolver provider is required")
            validate_fingerprint(resolver.fingerprint)
            if provider in providers:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            providers[provider] = resolver
        self._resolvers: "Mapping[str, CapabilityResolver]" = MappingProxyType(providers)

    def get(self, provider: str) -> "CapabilityResolver | None":
        return self._resolvers.get(provider)


__all__ = ["CapabilityResolverRegistry"]
