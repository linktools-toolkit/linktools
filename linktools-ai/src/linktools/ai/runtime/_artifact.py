#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact query and download API."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_json_bytes,
    canonical_sha256,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from .service_api import ArtifactDownload, ArtifactView
from .state._contracts import ArtifactState

_logger = environ.get_logger("ai.runtime.artifact")
_CURSOR_RESOURCE_KIND = "ARTIFACT"


class DefaultArtifactService:
    """Authorize metadata access before issuing an opaque download URL."""

    def __init__(self, state: ArtifactState, authorization: AuthorizationPolicy, *, grant_key: bytes, cursor_signer: CursorSigner, entry_path: str = "/v1/artifacts") -> None:
        if not grant_key:
            raise ValueError("artifact grant key is required")
        self._state = state
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
        page = await self._state.records.list_by_execution(
            execution_id,
            tenant_id=principal.tenant_id,
            cursor=raw_cursor,
            limit=validate_page_limit(limit),
        )
        values = tuple(ArtifactView(item.artifact_id, item.execution_id, item.size) for item in page.items)
        next_cursor = (
            None
            if page.next_cursor is None
            else encode_runtime_cursor(
                self._cursor_signer,
                tenant_id=principal.tenant_id,
                resource_kind=_CURSOR_RESOURCE_KIND,
                filter_digest=_artifact_filter(execution_id),
                position=page.next_cursor,
            )
        )
        return Page(values, next_cursor)

    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload:
        header = await self._state.records.get_header(artifact_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.ARTIFACT_READ, header)
        record = await self._state.records.get_metadata(artifact_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        expires_at = int(time.time()) + 300
        nonce = secrets.token_hex(16)
        request_digest = canonical_sha256({"action": "artifact.download", "tenant_id": principal.tenant_id, "principal_id": principal.principal_id, "artifact_id": artifact_id, "artifact_digest": record.digest})
        now = datetime.now(timezone.utc)
        operation = await self._state.operations.append(OperationLedgerInput(nonce, principal.tenant_id, ResourceKind.DOWNLOAD_GRANT, artifact_id, record.execution_id, OperationKind.DOWNLOAD_GRANT, OperationStatus.PENDING, request_digest, record.object_ref.key, record.digest, None, True, now, now))
        payload = {"tenant_id": principal.tenant_id, "principal_id": principal.principal_id, "artifact_id": artifact_id, "artifact_digest": record.digest, "expires_at": expires_at, "nonce": nonce}
        signature = hmac.new(self._grant_key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
        token = _encode_grant({**payload, "hmac": signature})
        await self._state.operations.compare_and_swap(nonce, tenant_id=principal.tenant_id, expected_status=OperationStatus.PENDING, next_record=OperationLedgerRecord(operation.operation_id, operation.tenant_id, operation.resource_kind, operation.resource_id, operation.execution_id, operation.operation_kind, OperationStatus.SUCCEEDED, operation.request_digest, record.object_ref.key, record.digest, None, operation.compactable, operation.sequence, operation.created_at, datetime.now(timezone.utc)))
        _logger.info("artifact grant issued: artifact=%s tenant=%s", artifact_id, principal.tenant_id)
        return ArtifactDownload(artifact_id, f"{self._entry_path}/{artifact_id}/download?grant={token}", str(expires_at))

    async def verify_grant(self, token: str, *, principal: Principal) -> str:
        try:
            payload = _decode_grant(token)
            signature = str(payload.pop("hmac"))
            expected = hmac.new(
                self._grant_key,
                canonical_json_bytes(payload),
                hashlib.sha256,
            ).hexdigest()
            if (
                not hmac.compare_digest(signature, expected)
                or str(payload["tenant_id"]) != principal.tenant_id
                or str(payload["principal_id"]) != principal.principal_id
            ):
                raise ValueError("invalid artifact grant")
            operation = await self._state.operations.get(
                str(payload["nonce"]),
                tenant_id=principal.tenant_id,
            )
            artifact_id = str(payload["artifact_id"])
            artifact_digest = str(payload["artifact_digest"])
            request_digest = canonical_sha256(
                {
                    "action": "artifact.download",
                    "tenant_id": principal.tenant_id,
                    "principal_id": principal.principal_id,
                    "artifact_id": artifact_id,
                    "artifact_digest": artifact_digest,
                }
            )
            expires_at = min(
                int(payload["expires_at"]),
                int(operation.created_at.timestamp()) + 300
                if operation is not None
                else 0,
            )
            if (
                operation is None
                or operation.status is not OperationStatus.SUCCEEDED
                or operation.result_digest != artifact_digest
                or operation.request_digest != request_digest
                or expires_at < int(time.time())
            ):
                raise ValueError("unknown artifact grant")
            record = await self._state.records.get_metadata(
                artifact_id,
                tenant_id=principal.tenant_id,
            )
            if record is None or record.digest != artifact_digest:
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
        raise ValueError("grant payload must be an object")  # noqa: TRY004
    return value


__all__ = ["DefaultArtifactService"]


def _artifact_filter(execution_id: str) -> str:
    return canonical_sha256({"execution_id": execution_id})


def _decode_cursor(cursor: str | None, tenant_id: str, execution_id: str, signer: CursorSigner) -> str | None:
    if cursor is None:
        return None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind=_CURSOR_RESOURCE_KIND,
        filter_digest=_artifact_filter(execution_id),
    )
    if payload.revision != 0:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return payload.position
