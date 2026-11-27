#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture retries retain the original canonical input."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, StructuredContent, TextContent

from linktools.ai.core import ExecutionContext, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import binary_attachment_text
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task.engine import TaskEngine


@pytest.mark.asyncio
@pytest.mark.parametrize('with_binary,matching_label', [(False, False), (True, False), (True, True)],
                         ids=['text', 'binary', 'structured-descriptor'])
async def test_agent_capture_retry_preserves_input_identity(
    tmp_path: Path, with_binary: bool, matching_label: bool,
) -> None:
    principal = service_principal('capture-identity', 'owner')
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open('capture-identity', storage=storage, tenant_id=principal.tenant_id) as persistence:
        async with TaskEngine.open(persistence=persistence, context=ExecutionContext(None)) as engine:
            inputs = engine.inputs
            suffix = (BinaryContent(b'body', media_type='text/plain'),) if with_binary else ()
            label = binary_attachment_text(suffix[0]) if matching_label else TextContent('fixed')
            prompt = (label, *suffix)
            ref = await inputs.create_agent_input(prompt, principal=principal, idempotency_key='capture')
            assert await inputs.create_agent_input(prompt, principal=principal, idempotency_key='capture') == ref
            with pytest.raises(AIError) as raised:
                await inputs.create_agent_input(
                    (StructuredContent({'value': 1}, content=label.content, tag='state', metadata=label.metadata), *suffix),
                    principal=principal, idempotency_key='capture',
                )
            assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            restored = await inputs.read_agent(ref, principal=principal)
            assert isinstance(restored.prompt[0], TextContent)
            assert not isinstance(restored.prompt[0], StructuredContent)

@pytest.mark.asyncio
async def test_new_capture_keeps_full_structured_identity() -> None:
    principal = service_principal('capture-identity', 'owner')
    async with Persistence.open('new-capture-identity', storage=PersistenceStorage.in_memory(), tenant_id=principal.tenant_id) as persistence:
        async with TaskEngine.open(persistence=persistence, context=ExecutionContext(None)) as engine:
            prompt = (StructuredContent({'value': 1}, content='fixed', tag='state'),)
            ref = await engine.inputs.create_agent_input(prompt, principal=principal, idempotency_key='current-capture')
            assert await engine.inputs.create_agent_input(prompt, principal=principal, idempotency_key='current-capture') == ref
            restored = await engine.inputs.read_agent(ref, principal=principal)
            assert isinstance(restored.prompt[0], StructuredContent)
            assert restored.prompt[0].value == {'value': 1}
            for changed in (
                StructuredContent({'value': 2}, content='fixed', tag='state'),
                StructuredContent({'value': True}, content='fixed', tag='state'),
                StructuredContent({'value': 1.0}, content='fixed', tag='state'),
                StructuredContent({'value': 1}, content='fixed', tag='other'),
                TextContent('fixed'),
            ):
                with pytest.raises(AIError) as raised:
                    await engine.inputs.create_agent_input((changed,), principal=principal, idempotency_key='current-capture')
                assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_evaluation_case_capture_preserves_structured_identity() -> None:
    from linktools.ai.evaluation import CaseRef, CaseSpec, DatasetRef, DatasetSpec
    from linktools.ai.evaluation.engine import Evaluations

    principal = service_principal('capture-evaluation', 'owner')
    async with (
        Persistence.open('evaluation-structured-identity', storage=PersistenceStorage.in_memory(), tenant_id=principal.tenant_id) as persistence,
        TaskEngine.open(persistence=persistence, context=ExecutionContext(None)) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        def spec(value: int) -> DatasetSpec:
            return DatasetSpec(DatasetRef('structured', 1), (
                CaseSpec.agent(CaseRef('structured', 'case', 1), prompt=(
                    StructuredContent({'value': value}, content='fixed', tag='state'),
                )),
            ))

        reference = await evaluations.publish_dataset(spec(1), principal=principal, idempotency_key='dataset-key')
        assert await evaluations.publish_dataset(spec(1), principal=principal, idempotency_key='dataset-key') == reference
        cases = await evaluations.list_cases(reference, principal=principal)
        captured = await engine.inputs.read_agent(cases.items[0].input, principal=principal)
        assert isinstance(captured.prompt[0], StructuredContent)
        assert captured.prompt[0].value == {'value': 1}
        with pytest.raises(AIError) as raised:
            await evaluations.publish_dataset(spec(2), principal=principal, idempotency_key='dataset-key')
        assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
