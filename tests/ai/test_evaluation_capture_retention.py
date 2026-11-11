#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State-owned input cleanup remains recoverable across interrupted deletion."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.captures import input_capture_state_dependency, read_capture_record
from linktools.ai.task.engine import TaskEngine
from .test_evaluation_consumers import CONTEXT, PRINCIPAL
from .test_evaluation_persistence import experiment
from .test_evaluation_retention import Offline
from .test_task_input_capture_planning import _source_capture


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ("before_delete", "after_delete", "before_acknowledgement"))
async def test_capture_cleanup_receipt_replays_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str,
) -> None:
    namespace = "capture-retention"
    now = datetime.now(timezone.utc)
    future = now + timedelta(minutes=2)
    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(persistence=persistence, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        source = await _source_capture(engine)
        contract = await engine.inputs.resolve_task_input(source, principal=PRINCIPAL)
        derived = await engine.inputs.create_task_input(contract, principal=PRINCIPAL,
            source_capture=source, idempotency_key="evaluation:expired")
        capture_key, _ = input_capture_state_dependency(derived)
        record = experiment()
        manifest = replace(record.manifest, principal=PRINCIPAL)
        record = replace(record, manifest=manifest, manifest_digest=manifest.digest,
            owned_input_captures=(derived,), content_expires_at=now + timedelta(minutes=1),
            metadata_expires_at=now + timedelta(minutes=1))
        await persistence.evaluation.records.reserve_experiment(record)

        with monkeypatch.context() as patched:
            maintenance = evaluations._retention._maintenance
            original_purge = maintenance.purge_unreferenced_captures

            async def interrupted_purge(*args, **kwargs):
                if interruption == "before_delete":
                    raise OSError("capture state is unavailable")
                await original_purge(*args, **kwargs)
                raise asyncio.CancelledError

            async def interrupted_acknowledgement(*args, **kwargs):
                raise OSError("cleanup acknowledgement is unavailable")

            if interruption == "before_acknowledgement":
                patched.setattr(persistence.evaluation.records, "acknowledge_cleanup", interrupted_acknowledgement)
            else:
                patched.setattr(maintenance, "purge_unreferenced_captures", interrupted_purge)
            with pytest.raises(asyncio.CancelledError if interruption == "after_delete" else OSError):
                await evaluations.purge_expired(principal=PRINCIPAL, now=future, exclusive=Offline())

        receipts = await persistence.evaluation.records.pending_cleanup(owner_principal_id=PRINCIPAL.principal_id, limit=None)
        assert len(receipts) == 1
        assert receipts[0].capture_keys == (capture_key,)
        assert receipts[0].objects == ()
        captured = await read_capture_record(persistence.task.tasks.state_store,
            namespace=namespace, tenant_id=PRINCIPAL.tenant_id, key=capture_key)
        assert (captured is not None) is (interruption == "before_delete")

    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(persistence=persistence, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        for unauthorized in (Principal("another-owner", PRINCIPAL.tenant_id, "service"),
                             Principal(PRINCIPAL.principal_id, "another-tenant", "service")):
            if unauthorized.tenant_id != PRINCIPAL.tenant_id:
                with pytest.raises(AIError) as denied:
                    await evaluations.purge_expired(principal=unauthorized, now=future, exclusive=Offline())
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            else:
                await evaluations.purge_expired(principal=unauthorized, now=future, exclusive=Offline())
            assert await persistence.evaluation.records.pending_cleanup(
                owner_principal_id=PRINCIPAL.principal_id, limit=None)
        await evaluations.purge_expired(principal=PRINCIPAL, now=future, exclusive=Offline())
        assert await persistence.evaluation.records.pending_cleanup(owner_principal_id=PRINCIPAL.principal_id, limit=None) == ()
        with pytest.raises(AIError) as expired:
            await engine.inputs.create_task_input(contract, principal=PRINCIPAL,
                source_capture=source, idempotency_key="evaluation:expired")
        assert expired.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        assert await engine.inputs.read_dependency(source, "value", principal=PRINCIPAL) == "accepted dependency"
        await evaluations.purge_expired(principal=PRINCIPAL, now=future + timedelta(seconds=1), exclusive=Offline())


@pytest.mark.asyncio
async def test_expired_capture_attachment_receipt_survives_missing_body_and_object_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.core import principal_identity_payload
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.captures import (
        capture_record_value, input_capture_expiry_key, input_capture_key, publish_capture_record,
    )
    from linktools.ai.persistence.codec import encode_domain
    from .test_evaluation_retention import CheckedObjects, put

    namespace = "expired-capture-attachments"
    now = datetime.now(timezone.utc)
    guard = Offline()
    objects = CheckedObjects(guard)
    plan = PersistenceStorage.from_root(tmp_path, exclusive_writer=True).plan
    capture_key = input_capture_key(namespace, PRINCIPAL.tenant_id, "agent", "expired")
    marker_key = input_capture_expiry_key(capture_key)
    reference = await put(objects, "capture-attachment", "private attachment")

    async def reject_listing():
        raise AssertionError("capture expiry must not enumerate the object bucket")
        yield

    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_plan(plan, object_store=objects),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(persistence=persistence, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        # The content deletion committed before its external attachment deletion.
        await publish_capture_record(persistence.task.tasks.state_store,
            namespace=namespace, tenant_id=PRINCIPAL.tenant_id, key=marker_key, value={
                "namespace": namespace, "tenant_id": PRINCIPAL.tenant_id,
                "capture_key": capture_key, "expired_at": now.isoformat(),
                "principal": principal_identity_payload(PRINCIPAL),
                "cleanup_objects": encode_domain(((PersistenceDomain.TASK, reference),)),
            })
        with monkeypatch.context() as without_listing:
            without_listing.setattr(objects, "list_objects", reject_listing)
            candidates = await engine.inputs.expire_inputs(principal=PRINCIPAL, now=now)
            assert candidates == (capture_key,)
            attachments = await evaluations._retention._maintenance.capture_objects(candidates)
            assert tuple(item.key for _, item in attachments) == (reference.key,)
            missing = await evaluations._retention._maintenance.purge_unreferenced_captures(candidates, exclusive=guard)
            assert missing.missing == (capture_key,)
        with monkeypatch.context() as patched:
            async def fail_delete(key: str, *, expected_digest: str) -> bool:
                raise OSError("object deletion is unavailable")

            patched.setattr(objects, "delete_object", fail_delete)
            with pytest.raises(OSError):
                await evaluations.purge_expired(principal=PRINCIPAL, now=now, exclusive=guard)
        assert await objects.stat(reference.key) is not None
        marker = await read_capture_record(persistence.task.tasks.state_store,
            namespace=namespace, tenant_id=PRINCIPAL.tenant_id, key=marker_key)
        assert marker is not None
        assert "cleanup_objects" in capture_record_value(marker)

    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_plan(plan, object_store=objects),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(persistence=persistence, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        result = await evaluations.purge_expired(principal=PRINCIPAL, now=now, exclusive=guard)
        assert result.objects_deleted == 1
        assert await objects.stat(reference.key) is None
        marker = await read_capture_record(persistence.task.tasks.state_store,
            namespace=namespace, tenant_id=PRINCIPAL.tenant_id, key=marker_key)
        assert marker is not None
        assert "cleanup_objects" not in capture_record_value(marker)
        assert await engine.inputs.expire_inputs(principal=PRINCIPAL, now=now) == ()
