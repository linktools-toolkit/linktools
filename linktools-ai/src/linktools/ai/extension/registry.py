#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit extension resolution and frozen feature registry."""

from ..domain.extension import Extension, ExtensionProvider, ExtensionResolution
from ..foundation.errors import ErrorCode, LinktoolsAIError


class FeatureRegistry:
    """Registration is open only during build/startup, then immutable."""

    def __init__(self) -> None:
        self._features: "dict[str, object]" = {}
        self._frozen = False

    def register(self, feature_id: str, value: object) -> None:
        if self._frozen:
            raise LinktoolsAIError(ErrorCode.FEATURE_REGISTRY_FROZEN, "feature registry is frozen")
        if feature_id in self._features and self._features[feature_id] != value:
            raise ValueError("feature already registered with another value")
        self._features[feature_id] = value

    def freeze(self) -> None:
        self._frozen = True

    def resolve(self, feature_id: str) -> object:
        return self._features[feature_id]

    @property
    def frozen(self) -> bool:
        return self._frozen


class ExtensionRegistry:
    """Resolve declarations from explicitly supplied providers."""

    def __init__(self, providers: "tuple[ExtensionProvider, ...]" = ()) -> None:
        self._providers = providers
        self._extensions = {extension.extension_id: extension for provider in providers for extension in provider.extensions}

    def register(self, provider: ExtensionProvider) -> None:
        if provider.provider_id in {item.provider_id for item in self._providers}:
            raise ValueError("extension provider already registered")
        self._providers = (*self._providers, provider)
        self._extensions.update({extension.extension_id: extension for extension in provider.extensions})

    def resolve(self, extension_id: str) -> ExtensionResolution:
        extension = self._extensions[extension_id]
        return ExtensionResolution(extension_id=extension.extension_id, provider_id=extension.provider_id, capability_ids=extension.capabilities)


__all__ = ["ExtensionRegistry", "FeatureRegistry"]
