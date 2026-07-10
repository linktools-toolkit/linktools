#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Middleware: a base class with no-op defaults for every lifecycle hook --
a concrete Middleware overrides only what it needs.

before_run/after_run/on_error are called directly by AgentRunner around its
agent.run(...) call (pydantic-ai has no equivalent native hook). before_model/
after_model/before_tool/after_tool get adapted into a real pydantic-ai
AbstractCapability by build_middleware_capability() (tool/executor.py
and AgentCompiler wire this up) -- this file only defines the
Protocol-like base class, it does not touch pydantic-ai at all."""

from typing import Any


class Middleware:
    async def before_run(self, context: Any) -> None:
        return None

    async def before_model(self, context: Any, request: Any) -> Any:
        return request

    async def before_tool(self, context: Any, request: Any) -> Any:
        return request

    async def after_tool(self, context: Any, request: Any, result: Any) -> Any:
        return result

    async def after_model(self, context: Any, response: Any) -> Any:
        return response

    async def after_run(self, context: Any, result: Any) -> Any:
        return result

    async def on_error(self, context: Any, error: Exception) -> None:
        return None
