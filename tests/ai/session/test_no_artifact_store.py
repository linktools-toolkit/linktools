#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses

from linktools.ai.session.types import FileSession, FileSessionSpec


def test_file_session_spec_has_no_artifact_store_field():
    field_names = {f.name for f in dataclasses.fields(FileSessionSpec)}
    assert "artifact_store" not in field_names


def test_file_session_has_no_artifact_store_field():
    field_names = {f.name for f in dataclasses.fields(FileSession)}
    assert "artifact_store" not in field_names


def test_file_session_persist_writes_call_sidecar(tmp_path):
    import asyncio
    from linktools.ai.core.model_runtime import RuntimeModelConfig
    from linktools.ai.session.types import SessionTurn

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    session = FileSession.create(tmp_path, FileSessionSpec(session_id="sidecar-test"))
    all_messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    turn = SessionTurn(
        history=[],
        all_messages=all_messages,
        model=RuntimeModelConfig(
            model_type="standard", protocol="openai", model="m", base_url="https://x",
            api_key="k", auth_token=None, timeout_seconds=300, raw={},
        ),
        token_usage={},
        llm_call={"call_id": "call-1"},
    )

    asyncio.run(session.persist(turn))

    sidecar_files = list((tmp_path / "calls").glob("*.json"))
    assert len(sidecar_files) == 1
