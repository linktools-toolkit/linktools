#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extension resolution service."""

class ExtensionService:
    def __init__(self, extensions: object, features: object) -> None:
        self._extensions = extensions
        self._features = features

    def resolve(self, extension_id: str) -> object:
        return self._extensions.resolve(extension_id)


__all__ = ["ExtensionService"]
