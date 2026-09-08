#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery-domain persistence for model attachment exposure facts."""

from collections.abc import Sequence

from ...core import canonical_sha256
from ...errors import AIError, ErrorCode
from ._attachments import (
    ModelExposure,
    ModelExposureEntry,
    model_exposure_activation_digest,
)
from ._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
    wire_type_id,
)
from ._plan import RuntimeDomain
from ._relocation import PathOrigin
from ._store import (
    FactQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    record_key_digest,
    stream_digest,
)


class ModelExposureRepository:
    """Own immutable model exposure facts for one Runtime namespace and tenant."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id

    @property
    def state_store(self) -> StateStore:
        return self._store

    def exposure_id(
        self,
        execution_id: str,
        step_run_id: str,
        run_step: int,
    ) -> str:
        _require_identity(execution_id, "execution_id")
        _require_identity(step_run_id, "step_run_id")
        _require_run_step(run_step)
        return canonical_sha256(
            [
                self._namespace,
                self._tenant_id,
                execution_id,
                step_run_id,
                run_step,
            ]
        )

    async def put(
        self,
        *,
        execution_id: str,
        step_run_id: str,
        run_step: int,
        path_origin: PathOrigin,
        entries: Sequence[ModelExposureEntry],
    ) -> ModelExposure:
        exposure_id = self.exposure_id(execution_id, step_run_id, run_step)
        frozen_entries = tuple(entries)
        candidate = ModelExposure(
            1,
            exposure_id,
            execution_id,
            step_run_id,
            run_step,
            path_origin,
            frozen_entries,
            model_exposure_activation_digest(frozen_entries),
        )
        stream = self._stream(exposure_id)
        subject = bytes.fromhex(exposure_id)
        owner = self._owner(execution_id)

        async def mutate(transaction: StateTransaction) -> ModelExposure:
            existing = await self._read_in_transaction(
                transaction,
                stream=stream,
                subject=subject,
                owner=owner,
                exposure_id=exposure_id,
            )
            if existing is not None:
                if existing != candidate:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return existing
            await transaction.insert_fact(
                StoredFact(
                    stream,
                    1,
                    owner,
                    "model_exposure",
                    subject,
                    None,
                    encode_envelope(
                        {
                            "type": wire_type_id(candidate),
                            "payload": _encode_persisted_domain(candidate),
                        }
                    ),
                )
            )
            return candidate

        try:
            return await self._store.mutate(mutate)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
        current = await self.get(
            execution_id=execution_id,
            step_run_id=step_run_id,
            run_step=run_step,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current != candidate:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return current

    async def get(
        self,
        *,
        execution_id: str,
        step_run_id: str,
        run_step: int,
    ) -> ModelExposure | None:
        exposure_id = self.exposure_id(execution_id, step_run_id, run_step)
        stream = self._stream(exposure_id)
        subject = bytes.fromhex(exposure_id)
        owner = self._owner(execution_id)
        return await self._store.read(
            lambda transaction: self._read_in_transaction(
                transaction,
                stream=stream,
                subject=subject,
                owner=owner,
                exposure_id=exposure_id,
            )
        )

    async def _read_in_transaction(
        self,
        transaction: StateTransaction,
        *,
        stream: bytes,
        subject: bytes,
        owner: bytes,
        exposure_id: str,
    ) -> ModelExposure | None:
        facts = await transaction.list_facts(
            FactQuery(
                stream_digest=stream,
                subject_digest=subject,
                latest=True,
            )
        )
        if not facts:
            return None
        if len(facts) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fact = facts[0]
        if (
            fact.stream_digest != stream
            or fact.sequence != 1
            or fact.owner_key_digest != owner
            or fact.kind != "model_exposure"
            or fact.subject_digest != subject
            or fact.state is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _decode_enveloped_domain(fact.data, ModelExposure)
        if (
            value.exposure_id != exposure_id
            or bytes.fromhex(value.exposure_id) != subject
            or self._owner(value.execution_id) != owner
            or self._stream(value.exposure_id) != stream
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def _stream(self, exposure_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            "model_exposure",
            exposure_id,
        )

    def _owner(self, execution_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.EXECUTION.value,
            "execution",
            execution_id,
        )


def _require_identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    return value


def _require_run_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("run_step must be a non-negative integer")
    return value


__all__ = ["ModelExposureRepository"]
