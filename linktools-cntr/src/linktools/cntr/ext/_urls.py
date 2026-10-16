#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy URL references exported through the extension package."""
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from linktools import utils
from linktools.runtime import lazy_load

if TYPE_CHECKING:
    from linktools.core import ConfigValue
    from linktools.runtime import Proxy
    from linktools.types import ConfigKeyType, QueryType
    from ..container import BaseContainer
    from ._base import Integration
    from ._nginx import Nginx


def load_config_url(container: "BaseContainer", key: "ConfigKeyType",
                    *path: str, queries: "QueryType | ConfigValue | None" = None) -> "Proxy":
    def make_url() -> str:
        url = container.get_config(key, type=str, default=None)
        if url:
            return utils.join_url(url, *path, queries=container.env_config.resolve(queries))
        return ""

    return lazy_load(make_url)


def load_port_url(container: "BaseContainer", key: "ConfigKeyType",
                  *path: str, queries: "QueryType | ConfigValue | None" = None,
                  https: bool = True) -> "Proxy":
    def make_url() -> str:
        port = container.get_config(key, type=int, default=0)
        if 0 < port <= 65535:
            return utils.make_url(
                "https" if https else "http",
                container.host,
                port,
                *path,
                queries=container.env_config.resolve(queries))
        return ""

    return lazy_load(make_url)


def load_nginx_url(container: "BaseContainer",
                   reference: "Nginx | tuple[Integration, ...] | None" = None, *path: str,
                   queries: "QueryType | ConfigValue | None" = None) -> "Proxy":
    from ._nginx_site import ResolvedSite

    def make_url() -> str:
        site = ResolvedSite.find(container, reference)
        url = site.get_url(default="")
        return utils.join_url(url, *path, queries=container.env_config.resolve(queries)) if url else ""

    return lazy_load(make_url)


def is_complete_url(value: str) -> bool:
    """Recognize complete URI targets without restricting independent schemes."""
    if "{url}" in value:
        return False
    try:
        parsed = urlsplit(value)
        if not parsed.scheme:
            return False
        if parsed.scheme in ("http", "https"):
            if not parsed.hostname:
                return False
            port = parsed.port
            return port is None or 0 < port <= 65535
        return True
    except ValueError:
        return False


def resolve_url_reference(value: str, base_url: str) -> str:
    """Resolve one explicit URL reference without formatting unrelated data."""
    if not value:
        return ""
    value = str(value)
    if value.startswith("{url}"):
        base = str(base_url)
        return base + value[len("{url}"):] if base else ""
    if urlsplit(value).scheme:
        return value
    base = str(base_url)
    return urljoin(base, value) if base else ""
