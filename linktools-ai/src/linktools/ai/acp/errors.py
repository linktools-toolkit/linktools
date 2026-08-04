#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP adapter errors and JSON-RPC mappings."""

from uuid import uuid4
from typing import Any, Mapping

from linktools.errors import ConfigError


class AcpDependencyError(ConfigError):
    """Raised when the optional ACP dependency is unavailable or mismatched."""


def require_sdk() -> object:
    try:
        import acp
    except ImportError as exc:
        raise AcpDependencyError(
            "ACP support requires agent-client-protocol==0.12.0; "
            "install linktools-ai[acp]"
        ) from exc
    from importlib.metadata import PackageNotFoundError, version as package_version

    try:
        version = package_version("agent-client-protocol")
    except PackageNotFoundError as exc:
        raise AcpDependencyError(
            "ACP support requires agent-client-protocol==0.12.0; install linktools-ai[acp]"
        ) from exc
    if version != "0.12.0":
        raise AcpDependencyError(
            f"ACP SDK version must be 0.12.0, found {version}; reinstall linktools-ai[acp]"
        )
    return acp


def request_error(
    reason: str,
    *,
    session_id: "str | None" = None,
    execution_id: "str | None" = None,
    details: "Mapping[str, Any] | None" = None,
) -> Exception:
    import acp

    data = {"errorId": uuid4().hex, "reason": reason}
    if session_id is not None:
        data["sessionId"] = session_id
    if execution_id is not None:
        data["executionId"] = execution_id
    if details:
        data.update(details)
    return acp.RequestError.invalid_params(data)


def internal_error(
    reason: str,
    *,
    session_id: "str | None" = None,
    execution_id: "str | None" = None,
    details: "Mapping[str, Any] | None" = None,
) -> Exception:
    import acp

    data = {"errorId": uuid4().hex, "reason": reason}
    if session_id is not None:
        data["sessionId"] = session_id
    if execution_id is not None:
        data["executionId"] = execution_id
    if details:
        data.update(details)
    return acp.RequestError.internal_error(data)


__all__ = ["AcpDependencyError", "internal_error", "request_error", "require_sdk"]
