#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP composition using the query-safe RuntimeAccess container."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..core import Principal
from ..errors import AIError, ErrorCode
from ._assembly import AppServices

REQUIRED_ROUTE_NAMES = frozenset({
    "execution",
    "session",
    "task",
    "approval",
    "event",
    "artifact",
    "evaluation",
})


class HttpHandler(Protocol):
    async def __call__(self, request: Mapping[str, str]) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class HttpRoute:
    name: str
    method: str
    path: str
    handler: HttpHandler

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.method.strip() or not self.path.strip() or not callable(self.handler):
            raise ValueError("HTTP route is incomplete")


class HttpApplication:
    def __init__(self, services: AppServices, routes: 'tuple[HttpRoute, ...]') -> None:
        route_names = {route.name for route in routes}
        if route_names != REQUIRED_ROUTE_NAMES:
            missing = sorted(REQUIRED_ROUTE_NAMES - route_names)
            extra = sorted(route_names - REQUIRED_ROUTE_NAMES)
            raise ValueError(f"HTTP route set is incomplete: missing={missing}, extra={extra}")
        if len(route_names) != len(routes):
            raise ValueError("HTTP route names must be unique")
        route_keys = {(route.method.upper(), route.path) for route in routes}
        if len(route_keys) != len(routes):
            raise ValueError("HTTP method and path pairs must be unique")
        if services.principal_provider is None:
            raise AIError(ErrorCode.SERVICE_NOT_READY, "HTTP authentication provider is not ready")
        self._access = services.access
        self._principal_provider = services.principal_provider
        self._routes = routes
        self._index = {(route.method.upper(), route.path): route.handler for route in routes}

    async def dispatch(self, method: str, path: str, request: 'Mapping[str, str]') -> 'Mapping[str, str]':
        key = (method.upper(), path)
        handler = self._index.get(key)
        if handler is None:
            raise AIError(
                ErrorCode.HTTP_ROUTE_NOT_FOUND,
                safe_details={"method": key[0], "path": path},
            )
        if self._principal_provider is None:
            raise AIError(ErrorCode.SERVICE_NOT_READY, "HTTP authentication provider is not ready")
        principal = await self._principal_provider.current()
        if not isinstance(principal, Principal):
            raise AIError(ErrorCode.SERVICE_NOT_READY, "HTTP authentication provider returned an invalid principal")
        authenticated = dict(request)
        authenticated["principal_id"] = principal.principal_id
        authenticated["tenant_id"] = principal.tenant_id
        authenticated["principal_kind"] = principal.kind
        return await handler(authenticated)


__all__ = ["HttpApplication", "HttpHandler", "HttpRoute", "REQUIRED_ROUTE_NAMES"]
