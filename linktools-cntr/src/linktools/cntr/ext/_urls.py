#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy URL references exported through the extension package."""
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from linktools import utils
from linktools.core import ConfigLazy

if TYPE_CHECKING:
    from linktools.core import Config, ConfigValue
    from linktools.types import ConfigKeyType, QueryType
    from ._nginx_site import ResolvedSite


def load_config_url(key: "ConfigKeyType",
                    *path: str, queries: "QueryType | ConfigValue | None" = None) -> "ConfigLazy":
    def make_url(config: "Config") -> str:
        url = config.get(key, type=str, default=None)
        if url:
            return utils.join_url(url, *path, queries=config.resolve(queries))
        return ""

    return ConfigLazy(make_url)


def load_port_url(key: "ConfigKeyType",
                  *path: str, queries: "QueryType | ConfigValue | None" = None,
                  https: bool = True) -> "ConfigLazy":
    def make_url(config: "Config") -> str:
        port = config.get(key, type=int, default=0)
        if 0 < port <= 65535:
            return utils.make_url(
                "https" if https else "http",
                config.get("HOST", type=str),
                port,
                *path,
                queries=config.resolve(queries))
        return ""

    return ConfigLazy(make_url)


def load_nginx_url(reference: "ResolvedSite | ConfigValue", *path: str,
                   queries: "QueryType | ConfigValue | None" = None) -> "ConfigLazy":
    def make_url(config: "Config") -> str:
        site = config.resolve(reference)
        url = site.get_url(default="")
        return utils.join_url(url, *path, queries=config.resolve(queries)) if url else ""

    return ConfigLazy(make_url)


def is_complete_url(value: str) -> bool:
    """Recognize complete URI targets without restricting independent schemes."""
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
    value = str(value).format(base_url=base_url)
    try:
        if urlsplit(value).scheme:
            return value
    except ValueError:
        return ""
    base = str(base_url)
    return urljoin(base, value) if base else ""
