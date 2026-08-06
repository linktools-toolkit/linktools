#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extension provider and resolver protocols."""

from typing import Protocol


class ExtensionProvider(Protocol):
    async def list(self) -> "tuple[object, ...]": ...
    async def resolve(self, extension_id: str) -> "object | None": ...


class ExtensionResolver(Protocol):
    def resolve(self, extension_id: str) -> object: ...


class FeatureRegistry(Protocol):
    def register(self, feature_id: str, value: object) -> None: ...
    def freeze(self) -> None: ...
    def resolve(self, feature_id: str) -> object: ...


__all__ = ["ExtensionProvider", "ExtensionResolver", "FeatureRegistry"]
