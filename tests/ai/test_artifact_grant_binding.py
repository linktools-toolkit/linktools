#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact grants stay bound to the durable authorization receipt."""

import hashlib
import hmac
from datetime import datetime, timezone
from typing import cast

import pytest

from linktools.ai.core import (
    AuthorizationAction,
    HmacCursorSigner,
    JsonValue,
    Principal,
    ResourceKind,
    TenantAuthorizationPolicy,
    canonical_json_bytes,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime import _artifact as artifact_module
from linktools.ai.runtime._artifact import (
    DefaultArtifactService,
    _decode_grant,
    _encode_grant,
)
from linktools.ai.runtime.state._contracts import ArtifactRecord
from linktools.ai.storage import ObjectRef


_GRANT_KEY = b"artifact-grant-test-key"


def _resign(payload: dict[str, str | int]) -> str:
    current = dict(payload)
    current.pop("hmac", None)
    signature = hmac.new(
        _GRANT_KEY,
        canonical_json_bytes(cast(JsonValue, current)),
        hashlib.sha256,
    ).hexdigest()
    return _encode_grant({**current, "hmac": signature})


class _RecordingAuthorization:
    def __init__(self) -> None:
        self.calls: list[tuple[AuthorizationAction, object]] = []

    async def authorize(
        self,
        principal: Principal,
        action: AuthorizationAction,
        resource: object,
    ) -> None:
        del principal
        self.calls.append((action, resource))


@pytest.mark.asyncio
async def test_artifact_list_authorizes_the_execution_identity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="artifact-list", tenant_id="tenant")
    try:
        await state.artifact.records.put_metadata(
            ArtifactRecord(
                artifact_id="artifact",
                execution_id="execution",
                tenant_id="tenant",
                producer="tool",
                media_type="text/plain",
                object_ref=ObjectRef("runtime", "artifact/key", "a" * 64, 7),
                created_at=datetime.now(timezone.utc),
            )
        )
        authorization = _RecordingAuthorization()
        service = DefaultArtifactService(
            state.artifact,
            authorization,  # type: ignore[arg-type]
            grant_key=_GRANT_KEY,
            cursor_signer=HmacCursorSigner("artifact", _GRANT_KEY),
        )

        page = await service.list(
            "execution",
            principal=Principal("caller", "tenant", "service"),
        )

        assert [item.artifact_id for item in page.items] == ["artifact"]
        assert len(authorization.calls) == 1
        action, resource = authorization.calls[0]
        assert action is AuthorizationAction.EXECUTION_READ
        assert resource.kind is ResourceKind.EXECUTION  # type: ignore[attr-defined]
        assert resource.resource_id == "execution"  # type: ignore[attr-defined]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_artifact_grant_is_bound_to_receipt_identity_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="artifact-grant", tenant_id="tenant")
    try:
        reference = ObjectRef(
            "runtime",
            "artifact/key",
            "a" * 64,
            7,
        )
        await state.artifact.records.put_metadata(
            ArtifactRecord(
                artifact_id="artifact",
                execution_id="execution",
                tenant_id="tenant",
                producer="tool",
                media_type="text/plain",
                object_ref=reference,
                created_at=datetime.now(timezone.utc),
            )
        )
        service = DefaultArtifactService(
            state.artifact,
            TenantAuthorizationPolicy("tenant"),
            grant_key=_GRANT_KEY,
            cursor_signer=HmacCursorSigner("artifact", _GRANT_KEY),
        )
        principal = Principal("caller", "tenant", "service")
        download = await service.get("artifact", principal=principal)
        token = download.url.split("grant=", 1)[1]

        assert await service.verify_grant(token, principal=principal) == reference.key

        payload = _decode_grant(token)
        nonce = str(payload["nonce"])
        operation = await state.artifact.operations.get(
            nonce,
            tenant_id=principal.tenant_id,
        )
        assert operation is not None

        wrong_principal = dict(payload)
        wrong_principal["principal_id"] = "other"
        with pytest.raises(AIError) as principal_error:
            await service.verify_grant(
                _resign(wrong_principal),
                principal=principal,
            )
        assert principal_error.value.code is ErrorCode.AUTHORIZATION_DENIED

        wrong_artifact = dict(payload)
        wrong_artifact["artifact_id"] = "other"
        with pytest.raises(AIError) as artifact_error:
            await service.verify_grant(
                _resign(wrong_artifact),
                principal=principal,
            )
        assert artifact_error.value.code is ErrorCode.AUTHORIZATION_DENIED

        wrong_digest = dict(payload)
        wrong_digest["artifact_digest"] = "b" * 64
        with pytest.raises(AIError) as digest_error:
            await service.verify_grant(
                _resign(wrong_digest),
                principal=principal,
            )
        assert digest_error.value.code is ErrorCode.AUTHORIZATION_DENIED

        extended = dict(payload)
        extended["expires_at"] = int(payload["expires_at"]) + 3600
        monkeypatch.setattr(
            artifact_module.time,
            "time",
            lambda: operation.created_at.timestamp() + 301,
        )
        with pytest.raises(AIError) as expiry_error:
            await service.verify_grant(
                _resign(extended),
                principal=principal,
            )
        assert expiry_error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()
