#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact query and download API."""

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorPayload,
    CursorSigner,
    OperationKind,
    OperationStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_json_bytes,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ._persistence import (
    OperationLedgerInput,
    OperationLedgerRecord,
    RuntimeDomainStates,
)
from ._services import ArtifactDownload, ArtifactView

_logger = environ.get_logger("ai.runtime.artifact")


class ArtifactApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ArtifactView]': ...
    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload: ...


class DefaultArtifactService:
    """Authorize metadata access before issuing an opaque download URL."""

    def __init__(self, persistence: RuntimeDomainStates, authorization: AuthorizationPolicy, *, grant_key: bytes, cursor_signer: CursorSigner, entry_path: str = "/v1/artifacts") -> None:
        if not grant_key:
            raise ValueError("artifact grant key is required")
        self._persistence = persistence
        self._authorization = authorization
        self._grant_key = grant_key
        self._cursor_signer = cursor_signer
        self._entry_path = entry_path.rstrip("/")

    async def list(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[ArtifactView]:
        await self._authorization.authorize(
            principal,
            AuthorizationAction.ARTIFACT_READ,
            ResourceRef(ResourceKind.ARTIFACT, execution_id, principal.tenant_id),
        )
        raw_cursor = _decode_cursor(cursor, principal.tenant_id, execution_id, self._cursor_signer)
        page = await self._persistence.artifact.records.list_by_execution(execution_id, tenant_id=principal.tenant_id, cursor=raw_cursor, limit=limit)
        values = tuple(ArtifactView(item.artifact_id, item.execution_id, item.size) for item in page.items)
        next_cursor = None if page.next_cursor is None else self._cursor_signer.encode(CursorPayload(1, principal.tenant_id, "ARTIFACT", _artifact_filter(execution_id), page.next_cursor, 0, int(time.time()) + 3600))
        return Page(values, next_cursor)

    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload:
        header = await self._persistence.artifact.records.get_header(artifact_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.ARTIFACT_READ, header)
        record = await self._persistence.artifact.records.get_metadata(artifact_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        expires_at = int(time.time()) + 300
        nonce = secrets.token_hex(16)
        request_digest = canonical_sha256({"action": "artifact.download", "tenant_id": principal.tenant_id, "principal_id": principal.principal_id, "artifact_id": artifact_id, "artifact_digest": record.digest})
        now = datetime.now(timezone.utc)
        operation = await self._persistence.artifact.operations.append(OperationLedgerInput(nonce, principal.tenant_id, ResourceKind.DOWNLOAD_GRANT, artifact_id, record.execution_id, OperationKind.DOWNLOAD_GRANT, OperationStatus.PENDING, request_digest, record.object_ref.key, record.digest, None, True, now, now))
        payload = {"tenant_id": principal.tenant_id, "principal_id": principal.principal_id, "artifact_id": artifact_id, "artifact_digest": record.digest, "expires_at": expires_at, "nonce": nonce}
        signature = hmac.new(self._grant_key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
        token = _encode_grant({**payload, "hmac": signature})
        await self._persistence.artifact.operations.compare_and_swap(nonce, tenant_id=principal.tenant_id, expected_status=OperationStatus.PENDING, next_record=OperationLedgerRecord(operation.operation_id, operation.tenant_id, operation.resource_kind, operation.resource_id, operation.execution_id, operation.operation_kind, OperationStatus.SUCCEEDED, operation.request_digest, record.object_ref.key, record.digest, None, operation.compactable, operation.sequence, operation.created_at, datetime.now(timezone.utc)))
        _logger.info("artifact grant issued: artifact=%s tenant=%s", artifact_id, principal.tenant_id)
        return ArtifactDownload(artifact_id, f"{self._entry_path}/{artifact_id}/download?grant={token}", str(expires_at))

    async def verify_grant(self, token: str, *, principal: Principal) -> str:
        try:
            payload = _decode_grant(token)
            signature = str(payload.pop("hmac"))
            expected = hmac.new(self._grant_key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected) or str(payload["tenant_id"]) != principal.tenant_id or str(payload["principal_id"]) != principal.principal_id or int(payload["expires_at"]) < int(time.time()):
                raise ValueError("invalid artifact grant")
            operation = await self._persistence.artifact.operations.get(str(payload["nonce"]), tenant_id=principal.tenant_id)
            if operation is None or operation.status is not OperationStatus.SUCCEEDED or operation.result_digest != str(payload["artifact_digest"]):
                raise ValueError("unknown artifact grant")
            record = await self._persistence.artifact.records.get_metadata(str(payload["artifact_id"]), tenant_id=principal.tenant_id)
            if record is None or record.digest != str(payload["artifact_digest"]):
                raise ValueError("artifact grant target mismatch")
            return record.object_ref.key
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error


def _encode_grant(payload: dict[str, str | int]) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).rstrip(b"=").decode("ascii")


def _decode_grant(token: str) -> dict[str, str | int]:
    padding = "=" * (-len(token) % 4)
    value = json.loads(base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("grant payload must be an object")
    return value


__all__ = ["ArtifactApi", "DefaultArtifactService"]


def _artifact_filter(execution_id: str) -> str:
    return canonical_sha256({"execution_id": execution_id})


def _decode_cursor(cursor: str | None, tenant_id: str, execution_id: str, signer: CursorSigner) -> str | None:
    if cursor is None:
        return None
    try:
        payload = signer.decode(cursor)
        if payload.cursor_version != 1 or payload.tenant_id != tenant_id or payload.resource_kind != "ARTIFACT" or payload.filter_digest != _artifact_filter(execution_id) or payload.snapshot_or_store_revision != 0 or not payload.sort_key.strip():
            raise ValueError("artifact cursor identity mismatch")
        return payload.sort_key
    except (binascii.Error, KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
