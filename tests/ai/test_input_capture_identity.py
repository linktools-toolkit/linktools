#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture retries retain the projection chosen by their immutable object owner."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, StructuredContent, TextContent

from linktools.ai.core import ExecutionContext, canonical_sha256, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import task_prompt_draft
from linktools.ai.model import MaterializedUserContent
from linktools.ai.model.messages import binary_attachment_text
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task.engine import TaskEngine


@pytest.mark.asyncio
@pytest.mark.parametrize('with_binary,matching_label', [(False, False), (True, False), (True, True)],
                         ids=['text', 'binary', 'structured-descriptor'])
async def test_legacy_agent_capture_retry_keeps_original_bytes(
    tmp_path: Path, with_binary: bool, matching_label: bool,
) -> None:
    principal = service_principal('capture-identity', 'owner')
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open('capture-identity', storage=storage, tenant_id=principal.tenant_id) as persistence:
        async with TaskEngine.open(persistence=persistence, context=ExecutionContext(None)) as engine:
            inputs = engine.inputs
            suffix = (BinaryContent(b'body', media_type='text/plain'),) if with_binary else ()
            label = binary_attachment_text(suffix[0]) if matching_label else TextContent('fixed')
            old_prompt = (label, *suffix)
            accepted = await inputs.materialize_input(old_prompt)
            payload = {
                'prompt': task_prompt_draft(accepted),
                'input_view': dict(accepted.view) if isinstance(accepted, MaterializedUserContent) else None,
                'binding': None, 'original_input': {}, 'source_execution_id': None,
                'source_invocation_id': None, 'repository_instructions': None,
                'task_input': None, 'source_principal': None, 'input_context': None,
            }
            identity, envelope = inputs._publication('agent', principal, 'legacy-capture', payload)
            envelope.pop('request_identity_version')
            key = inputs._key(principal.tenant_id, 'agent', identity)
            old_digest = await inputs._put(key, envelope)
            new_prompt = (StructuredContent({'value': 1}, content=label.content, tag='state', metadata=label.metadata), *suffix)
            replay = await inputs.create_agent_input(new_prompt, principal=principal, idempotency_key='legacy-capture')
            assert replay.digest == old_digest
            assert await inputs._get(key) == envelope
            assert canonical_sha256(envelope) == old_digest
            restored = await inputs.read_agent(replay, principal=principal)
            assert isinstance(restored.prompt[0], TextContent)
            assert not isinstance(restored.prompt[0], StructuredContent)
            with pytest.raises(AIError) as raised:
                await inputs.create_agent_input(
                    (StructuredContent({'value': 1}, content='changed', tag='state'), *suffix),
                    principal=principal, idempotency_key='legacy-capture',
                )
            assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


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
