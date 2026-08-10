#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment-owned model connection and credential contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class ModelConnectionConfig:
    connection_id: str
    base_url: "str | None" = None
    timeout_seconds: "float | None" = None
    credential_id: "str | None" = None

    def __post_init__(self) -> None:
        if not self.connection_id.strip():
            raise ValueError("model connection id is required")
        normalized = None if self.base_url is None else _normalize_base_url(self.base_url)
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("model connection timeout must be positive")
        if self.credential_id is not None and not self.credential_id.strip():
            raise ValueError("model connection credential id is required")
        object.__setattr__(self, "base_url", normalized)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "connection_id": self.connection_id,
                "base_url": self.base_url,
                "timeout_seconds": self.timeout_seconds,
                "credential_id": self.credential_id,
            }
        )


class ModelConnectionRegistry:
    def __init__(self, connections: Sequence[ModelConnectionConfig] = ()) -> None:
        values: dict[str, ModelConnectionConfig] = {}
        for connection in connections:
            previous = values.get(connection.connection_id)
            if previous is not None and previous != connection:
                raise AIError(ErrorCode.MODEL_CONNECTION_CONFLICT)
            values[connection.connection_id] = connection
        self._connections: "Mapping[str, ModelConnectionConfig]" = MappingProxyType(values)

    def resolve(self, connection_id: str) -> ModelConnectionConfig:
        try:
            return self._connections[connection_id]
        except KeyError as error:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND) from error

    def resolve_optional(self, connection_id: "str | None") -> "ModelConnectionConfig | None":
        return None if connection_id is None else self.resolve(connection_id)


class ModelCredentialProvider(Protocol):
    def get_api_key(self, credential_id: str) -> "str | None": ...


class StaticModelCredentialProvider(ModelCredentialProvider):
    def __init__(self, credentials: "Mapping[str, str] | None" = None) -> None:
        self._credentials = MappingProxyType(dict(credentials or {}))

    def get_api_key(self, credential_id: str) -> "str | None":
        return self._credentials.get(credential_id)


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("model connection base_url port is invalid") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("model connection base_url must be a clean absolute HTTP URL")
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))


__all__ = ["ModelConnectionConfig", "ModelConnectionRegistry", "ModelCredentialProvider", "StaticModelCredentialProvider"]
