#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Runtime tests and persistence fixtures."""

import asyncio
import os
import sqlite3
from pathlib import Path
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolReturnPart
from pydantic_ai.models import (
    CompletedStreamedResponse,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from linktools.ai.asset import AssetStore
from linktools.ai.capability import ShellCapability
from linktools.ai.capability import AssetCapability, FileSystem, tool_metadata
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.model import ModelBinding
from linktools.ai.core import JsonValue
from linktools.ai.capability import TraceCapability
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepPersistence, StepStore, SqliteStepStore
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.persistence.contracts import StoredUserInput
from linktools.ai.storage import StoredPayload


_ReadT = TypeVar("_ReadT")


def is_tool_retry(part: object) -> bool:
    """Recognize native retry feedback on the supported SDK message shapes."""
    return isinstance(part, ToolReturnPart) and part.outcome == "retried"


def model_features(
    binding: ModelBinding | Callable[[], ModelBinding],
    features: RuntimeFeatures | None = None,
    *,
    route_id: str = "default",
) -> RuntimeFeatures:
    """Declare an offline model on the feature set being exercised."""
    if features is None:
        features = RuntimeFeatures()
    features.model(route_id, binding)
    return features


def native_checkpoint(*, database: Path | None = None) -> StepPersistence:
    """Reopen a real official store shared only by this test's recovery attempts.

    Runtime run IDs include namespace, tenant and execution identity. Low-level
    tests with manually chosen run IDs can pass separate databases explicitly.
    The environment path is inherited by process-exit recovery probes.
    """
    if database is None:
        database = Path(os.environ["LINKTOOLS_TEST_STEP_DATABASE"])
    if not database.exists():
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
    return StepPersistence(store=SqliteStepStore(database=database), capture_frontier=True)


async def seed_native_snapshot(
    store: StepStore, *, run_id: str, messages: list[ModelMessage],
    step_index: int = 1, state: Literal["complete", "interrupted"] = "complete",
    conversation_id: str | None = None, agent_name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Seed official source records for tests that control a precise crash boundary."""
    if await store.get_run(run_id=run_id) is None:
        await store.register_run(RunRecord(
            run_id=run_id, conversation_id=conversation_id, agent_name=agent_name,
            metadata={"linktools.transcript_start": "0", **(metadata or {})},
        ))
    from linktools.ai.persistence.runs import NativeCheckpoint
    await store.save_snapshot(NativeCheckpoint.encode(ContinuableSnapshot(
        run_id=run_id, messages=messages, step_index=step_index, state=state,
        conversation_id=conversation_id, agent_name=agent_name, idempotency_key=uuid4().hex,
    ), output_retries_used=0))


def _persistence_capabilities(*, restart: bool = False, trace: bool = False) -> RuntimeFeatures:
    group = RuntimeFeatures("persistence")
    if restart:
        group.capability(native_checkpoint(), scope="runtime")
    if trace:
        group.capability(TraceCapability(), scope="runtime")
    return group


def workspace_features(
    group_id: str, *, workspace: Any,
    features_type: type[RuntimeFeatures] = RuntimeFeatures, assets: AssetStore | None = None, **kwargs: Any,
) -> RuntimeFeatures:
    features = features_type(group_id, **kwargs)
    if assets is not None:
        features.capability(AssetCapability(assets), scope="runtime")
    if workspace is not None:
        features.capability(FileSystem(workspace), scope="runtime")
        features.capability(ShellCapability(), scope="runtime")
    return features


class _Models:
    route_id = "default"
    provider = "test"
    model_identity = "test:live-history"
    model_digest = "a" * 64
    contract: dict[str, JsonValue] = {"provider": "test", "model": "live-history", "input_mime_types": ["application/*", "audio/*", "text/*", "video/*"]}

    def __init__(self, model: FunctionModel) -> None:
        self.model = model

    def materialize(self) -> FunctionModel:
        return self.model


async def _wait_for_committed(
    read: Callable[[], Awaitable[_ReadT]],
    ready: Callable[[_ReadT], bool],
    *,
    timeout: float = 5.0,
) -> _ReadT:
    """Wait for a canonical read condition without advancing the producer."""
    async def wait() -> _ReadT:
        while True:
            value = await read()
            if ready(value):
                return value
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(wait(), timeout=timeout)


def tool_run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


def execution_owner_fields(prompt: str = "prompt") -> dict[str, object]:
    return {
        "principal_id": "principal",
        "principal_kind": "service",
        "stored_user_input": StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
        ),
    }


def tool_with_metadata(
    function: Callable[..., Any],
    descriptor: ManagedToolDescriptor,
) -> Tool[Any]:
    path_fields = list(descriptor.workspace_path_fields) or None
    return Tool(
        function,
        metadata=tool_metadata(
            effect_policy=descriptor.effect_policy,
            tool_class=descriptor.tool_class,
            path_fields=path_fields,
        ),
    )


async def _runtime_usage_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del messages, info
    return ModelResponse(
        parts=[TextPart("done")],
        usage=RequestUsage(
            input_tokens=101,
            output_tokens=202,
            cache_read_tokens=303,
            cache_write_tokens=404,
        ),
    )


class _UsageFunctionModel(FunctionModel):
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        response = await self.request(
            messages,
            model_settings,
            model_request_parameters,
        )
        yield CompletedStreamedResponse(
            response,
            model_request_parameters=model_request_parameters,
            replay_events=True,
        )


class _RuntimeUsageModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:usage"
    model_digest = "u" * 64
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "usage",
    }

    def materialize(self) -> FunctionModel:
        return _UsageFunctionModel(_runtime_usage_model)


class RuntimeUsageModels(_RuntimeUsageModelBinding):
    pass


class FileToolsStub:
    """Complete bounded-file test double that rejects unexpected operations."""

    async def unsupported(self, *args: object, **kwargs: object) -> Any:
        raise NotImplementedError("Unexpected file operation in this test")

    run_command = start_command = check_command = stop_command = unsupported
    canonicalize_path = read_bytes = inspect_file = read_text = unsupported
    list_files = search_text = write_file = unsupported
