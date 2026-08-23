"""Ordered middleware lifecycle pipeline."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._scope import RunContext


class Middleware(Protocol):
    mutating: bool

    async def before_run(self, context: RunContext) -> None: ...
    async def before_model(self, context: RunContext) -> None: ...
    async def after_model(self, context: RunContext) -> None: ...
    async def before_tool(self, context: RunContext) -> None: ...
    async def after_tool(self, context: RunContext) -> None: ...
    async def on_error(self, context: RunContext, error: BaseException) -> None: ...
    async def after_run(self, context: RunContext) -> None: ...


class MiddlewarePipeline:
    def __init__(self, middleware: 'Sequence[Middleware]' = ()) -> None:
        self._middleware = tuple(middleware)
        self._logger = environ.get_logger("ai.observe.middleware")

    @property
    def fingerprint(self) -> str:
        from ..core import canonical_sha256

        return canonical_sha256([type(item).__qualname__ for item in self._middleware])

    async def before_run(self, context: RunContext) -> None:
        await self._run_before(context, "before_run")

    async def before_model(self, context: RunContext) -> None:
        await self._run_before(context, "before_model")

    async def after_model(self, context: RunContext) -> None:
        await self._run_after(context, "after_model")

    async def before_tool(self, context: RunContext) -> None:
        await self._run_before(context, "before_tool")

    async def after_tool(self, context: RunContext) -> None:
        await self._run_after(context, "after_tool")

    async def on_error(self, context: RunContext, error: BaseException) -> None:
        for middleware in reversed(self._middleware):
            try:
                await middleware.on_error(context, error)
            except asyncio.CancelledError:
                raise
            except Exception as failure:  # noqa: BLE001
                self._handle_failure(middleware, "on_error", failure)

    async def after_run(self, context: RunContext) -> None:
        await self._run_after(context, "after_run")

    async def _run_before(self, context: RunContext, stage: str) -> None:
        for middleware in self._middleware:
            try:
                await self._dispatch(middleware, stage, context)
            except Exception as exc:
                if middleware.mutating:
                    raise AIError(ErrorCode.MIDDLEWARE_FAILED, f"middleware {stage} failed") from exc
                self._logger.warning(
                    "observational middleware failed: stage=%s",
                    stage,
                )

    async def _run_after(self, context: RunContext, stage: str) -> None:
        for middleware in reversed(self._middleware):
            try:
                await self._dispatch(middleware, stage, context)
            except Exception as exc:  # noqa: BLE001
                self._handle_failure(middleware, stage, exc)

    def _handle_failure(
        self,
        middleware: Middleware,
        stage: str,
        failure: BaseException,
    ) -> None:
        if middleware.mutating:
            raise AIError(ErrorCode.MIDDLEWARE_FAILED, f"middleware {stage} failed") from failure
        self._logger.warning(
            "observational middleware failed: stage=%s",
            stage,
        )

    async def _dispatch(self, middleware: Middleware, stage: str, context: RunContext) -> None:
        if stage == "before_run":
            await middleware.before_run(context)
        elif stage == "before_model":
            await middleware.before_model(context)
        elif stage == "after_model":
            await middleware.after_model(context)
        elif stage == "before_tool":
            await middleware.before_tool(context)
        elif stage == "after_tool":
            await middleware.after_tool(context)
        elif stage == "after_run":
            await middleware.after_run(context)
        else:
            raise ValueError(f"unsupported middleware stage: {stage}")

__all__ = ["Middleware", "MiddlewarePipeline"]
