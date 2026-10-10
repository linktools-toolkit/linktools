#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical executions retain Agent limits without a shared run budget."""

import base64
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import httpx
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime.state._codec import _decode_enveloped_domain
from linktools.ai.runtime.state._contracts import ExecutionRecord

from . import _runtime_test_helpers as helpers


_FIXTURE = Path(__file__).parent / "fixtures" / "persistence" / "sqlite_session_budget_4609177e.json"


def _restore_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    sql = zlib.decompress(base64.b64decode(fixture["sql_zlib_base64"]))
    assert hashlib.sha256(sql).hexdigest() == fixture["sql_sha256"]
    assert b"budget_scope_id" not in sql
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(sql.decode("utf-8"))
    return path, fixture["execution_id"], fixture["standalone_execution_id"]


@pytest.mark.asyncio
async def test_sqlite_execution_without_shared_budget_keeps_public_history(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    path, execution_id, _standalone_id = _restore_fixture(tmp_path)
    async with RuntimeHistory.open("older-history", storage=RuntimeStorage.sqlite(path)) as history:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(history=history)),
            base_url="http://127.0.0.1:8765",
        ) as client:
            for route in (
                "/api/sessions?limit=50",
                "/api/session?session_id=session&include_timeline=false",
                "/api/executions?session_id=session&limit=1",
                f"/api/executions/{execution_id}",
                f"/api/executions/{execution_id}/trace",
                f"/api/executions/{execution_id}/models",
            ):
                response = await client.get(route)
                assert response.status_code == 200, response.text
            response = await client.get("/api/session?session_id=session&limit=50")
            assert response.status_code == 200, response.text
            turn, = response.json()["timeline"]["items"]
            assert turn["execution_id"] == execution_id
            assert turn["user_input"]["prompt"] == {"kind": "text", "text": "retained older question"}
            assert turn["conversation_committed"] is True
            assert [(item["item_kind"], item["content"]) for item in turn["items"]] == [
                ("assistant", "done"),
            ]


@pytest.mark.asyncio
async def test_sqlite_retry_preserves_historical_agent_usage_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _session_execution_id, execution_id = _restore_fixture(tmp_path)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart("over limit")], usage=RequestUsage(output_tokens=303))

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    application = CapabilityGroup("application")
    application.agent("default", revision=2, model="default", allow_tools=())
    async with Runtime.open(
        "older-history", storage=RuntimeStorage.sqlite(path),
        models=helpers.RuntimeUsageModels(), capabilities=(application,),
    ) as runtime:
        original = await runtime.executions.get(execution_id)
        assert await original.budget_usage() is None
        retried = await original.retry("exceed the original output limit")
        result = (await retried.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        assert result.safe_error_details["limits"]["output_tokens"] == 250
        assert await retried.budget_usage() is None
        fresh = await runtime.agents.get().run("use the current unrestricted declaration", timeout_seconds=10)
        assert fresh.result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("scope", (None, "execution:owner", "graph:owner", "", 0, False))
def test_explicit_shared_budget_scope_keeps_its_value_and_validation(
    tmp_path: Path, scope: object,
) -> None:
    path, _execution_id, _standalone_id = _restore_fixture(tmp_path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT payload_json FROM ai_state_records WHERE kind = 'execution'",
        ).fetchone()
    value = json.loads(row[0])
    value["value"]["payload"]["fields"]["budget_scope_id"] = scope
    if scope is None or isinstance(scope, str) and scope:
        assert _decode_enveloped_domain(value, ExecutionRecord).budget_scope_id == scope
    else:
        with pytest.raises(AIError) as invalid:
            _decode_enveloped_domain(value, ExecutionRecord)
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
