#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Startup-pinned provider model registry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StartupModelRegistry:
    routes: "tuple[tuple[str, str], ...]"

    @classmethod
    def build(cls, routes: "dict[str, str]") -> "StartupModelRegistry":
        return cls(tuple(sorted(routes.items())))

    def resolve(self, route_id: str) -> str:
        return dict(self.routes)[route_id]


__all__ = ["StartupModelRegistry"]
