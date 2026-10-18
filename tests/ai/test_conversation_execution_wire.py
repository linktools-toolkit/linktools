#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution conversation promises are explicit in the durable wire."""

from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ConversationCapability
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import ExecutionRecord

from .test_explicit_runtime_features import _FeatureModels, _features


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
@pytest.mark.parametrize("conversation", (False, True), ids=("minimal", "conversation"))
async def test_execution_writer_requires_explicit_conversation(
    persisted: bool, conversation: bool,
) -> None:
    storage = RuntimeStorage.in_memory()
    features = _features(*(ConversationCapability(),) if conversation else ())
    async with Runtime.open(
        "conversation-wire", storage=storage, models=_FeatureModels(), features=features,
    ) as runtime:
        execution = await runtime.agents.get().start("answer")
        await execution.wait(timeout_seconds=5)
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.conversation is conversation
        encode = _encode_persisted_domain if persisted else encode_domain
        payload = encode(record)
        assert payload["fields"]["conversation"] is conversation

        def decode(value: object) -> ExecutionRecord:
            if persisted:
                return _decode_enveloped_domain(
                    encode_envelope({"type": "execution_record", "payload": value}), ExecutionRecord,
                )
            return decode_domain(value, ExecutionRecord)

        assert decode(payload) == record
        missing = {key: value for key, value in payload["fields"].items() if key != "conversation"}
        with pytest.raises(AIError) as rejected:
            decode({**payload, "fields": missing})
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        for invalid in (None, 0, "true"):
            with pytest.raises(TypeError, match="persistence guarantees"):
                replace(record, conversation=invalid)
            with pytest.raises(AIError) as rejected:
                decode({**payload, "fields": {**payload["fields"], "conversation": invalid}})
            assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
