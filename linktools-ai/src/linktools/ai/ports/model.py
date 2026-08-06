#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model plan resolution protocol."""

from typing import Protocol


class ModelRegistry(Protocol):
    def resolve_plan(self, request: object) -> object: ...


__all__ = ["ModelRegistry"]
