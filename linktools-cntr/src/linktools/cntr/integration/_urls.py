#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy URL references exported through the integration package."""
from typing import TYPE_CHECKING

from linktools import utils
from linktools.runtime import lazy_load

if TYPE_CHECKING:
    from linktools.runtime import Proxy
    from linktools.types import ConfigKeyType, QueryType
    from ..container import BaseContainer


def load_config_url(container: "BaseContainer", key: "ConfigKeyType",
                    *path: str, queries: "QueryType | None" = None) -> "Proxy":
    def make_url() -> str:
        url = container.get_config(key, type=str, default=None)
        if url:
            return utils.join_url(url, *path, queries=queries)
        return ""

    return lazy_load(make_url)


def load_port_url(container: "BaseContainer", key: "ConfigKeyType",
                  *path: str, queries: "QueryType | None" = None,
                  https: bool = True) -> "Proxy":
    def make_url() -> str:
        port = container.get_config(key, type=int, default=0)
        if 0 < port < 65535:
            return utils.make_url(
                "https" if https else "http",
                container.host,
                port,
                *path,
                queries=queries)
        return ""

    return lazy_load(make_url)


def load_nginx_url(container: "BaseContainer", local_id: str, *path: str,
                   queries: "QueryType | None" = None) -> "Proxy":
    from ..container import ContainerError
    if not isinstance(local_id, str) or not local_id:
        raise ContainerError("Nginx site ID must be a nonempty string")

    def make_url() -> str:
        site = container.manager.nginx_sites.get((container.name, local_id))
        if site is None:
            raise ContainerError("Unknown nginx site %s/%s" % (container.name, local_id))
        url = site.url
        return utils.join_url(url, *path, queries=queries) if url else ""

    return lazy_load(make_url)
