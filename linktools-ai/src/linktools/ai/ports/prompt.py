#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt snapshot metadata protocol."""

from typing import Protocol


class PromptRepository(Protocol):
    async def save(self, prompt: object) -> object: ...
    async def get(self, snapshot_id: str) -> "object | None": ...


__all__ = ["PromptRepository"]
