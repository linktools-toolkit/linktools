#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure navigation links and lazy nginx site URL references."""
from typing import TYPE_CHECKING

from linktools import utils
from linktools.core import LazyProvider
from linktools.runtime import lazy_load

if TYPE_CHECKING:
    from typing import Any
    from linktools.runtime import Proxy
    from linktools.types import ConfigKeyType, QueryType
    from ..container import BaseContainer


class ExposeCategory:

    def __init__(self, name: str, desc: str) -> None:
        self.name = name
        self.desc = desc

    def __call__(self, name: str, icon: str, desc: str, url: str) -> "ExposeLink":
        return ExposeLink(self, name, icon, desc or name, url)


class ExposeLink:

    def __init__(self, category: "ExposeCategory", name: str, icon: str, desc: str, url: str) -> None:
        self.category = category
        self.name = name
        self.icon = icon
        self.desc = desc
        self._url = url

    @property
    def url(self) -> "str | None":
        if not self._url:
            return None
        return str(self._url)

    @property
    def is_valid(self) -> bool:
        return not not self.url


class ExposeMixin:
    expose_public = ExposeCategory("public", "Public")
    expose_private = ExposeCategory("private", "Private")
    expose_container = ExposeCategory("container", "Internal")
    expose_other = ExposeCategory("other", "Tools")

    def load_config_url(self: "BaseContainer", key: "ConfigKeyType",
                        *path: str, queries: "QueryType | None" = None) -> "Proxy":
        def make_url() -> str:
            url = self.get_config(key, type=str, default=None)
            if url:
                return utils.join_url(url, *path, queries=queries)
            return ""

        return lazy_load(make_url)

    def load_port_url(self: "BaseContainer", key: "ConfigKeyType",
                      *path: str, queries: "QueryType | None" = None,
                      https: bool = True) -> "Proxy":
        def make_url() -> str:
            port = self.get_config(key, type=int, default=0)
            if 0 < port < 65535:
                return utils.make_url(
                    "https" if https else "http",
                    self.host,
                    port,
                    *path,
                    queries=queries)
            return ""

        return lazy_load(make_url)

    def load_nginx_url(self: "BaseContainer", local_id: str, *path: str,
                       queries: "QueryType | None" = None) -> "Proxy":
        from ..container import ContainerError
        if not isinstance(local_id, str) or not local_id:
            raise ContainerError("Nginx site ID must be a nonempty string")

        def make_url() -> str:
            site = self.manager.nginx_sites.get((self.name, local_id))
            if site is None:
                raise ContainerError("Unknown nginx site %s/%s" % (self.name, local_id))
            url = site.url
            return utils.join_url(url, *path, queries=queries) if url else ""

        return lazy_load(make_url)


class NginxMixin:

    def get_nginx_domain(self: "BaseContainer", name: "str | None" = None) -> "LazyProvider":

        def get_domain(cfg: "dict[str, Any]") -> str:
            if not self.containers["nginx"].enable:
                return ""
            if not cfg.get("NGINX_WILDCARD_DOMAIN", type=bool):
                return cfg.get("NGINX_ROOT_DOMAIN")
            root_domain = cfg.get("NGINX_ROOT_DOMAIN")
            if root_domain in ("_", "localhost"):
                return root_domain
            if name is None:
                return f"{self.name}.{root_domain}"
            elif name.strip() == "":
                return root_domain
            return f"{name}.{root_domain}"

        return LazyProvider(get_domain)
