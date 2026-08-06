#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extension provider protocol."""

from typing import Protocol


class ExtensionProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve(self, extension_id: str) -> str: ...


__all__ = ["ExtensionProvider"]
