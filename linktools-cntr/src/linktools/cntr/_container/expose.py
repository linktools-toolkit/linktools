#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Access-link declarations (exposes) for a container: categories, lazily
resolved URLs, and the idempotent nginx-conf registration hook."""
from copy import deepcopy
from typing import TYPE_CHECKING

from linktools import utils
from linktools.runtime import lazy_load
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import ConfigKeyType, PathType, QueryType
    from ..container import BaseContainer


def _freeze(value):
    """Recursively turn dict/list/set values into a hashable, deterministic
    tuple so they can be used inside a start-hook dedup key. set iteration
    order depends on insertion history, not just content, so sets are sorted
    by their frozen representation; list/tuple order is preserved as-is since
    it's semantically meaningful. Each container type is tagged so a list and
    a set with the same elements don't collide."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted((k, _freeze(v)) for k, v in value.items())))
    if isinstance(value, set):
        return ("set", tuple(sorted((_freeze(v) for v in value), key=repr)))
    if isinstance(value, (list, tuple)):
        return ("list" if isinstance(value, list) else "tuple", tuple(_freeze(v) for v in value))
    return value


class ExposeCategory:

    def __init__(self, name: str, desc: str):
        self.name = name
        self.desc = desc

    def __call__(self, name: str, icon: str, desc: str, url: str):
        return ExposeLink(self, name, icon, desc or name, url)


class ExposeLink:

    def __init__(self, category: "ExposeCategory", name: str, icon: str, desc: str, url: str):
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
                        *path: str, queries: "QueryType | None" = None):
        def make_url():
            url = self.get_config(key, type=str, default=None)
            if url:
                return utils.join_url(url, *path, queries=queries)
            return ""

        return lazy_load(make_url)

    def load_port_url(self: "BaseContainer", key: "ConfigKeyType",
                      *path: str, queries: "QueryType | None" = None,
                      https: bool = True):
        def make_url():
            port = self.get_config(key, type=int, default=0)
            if 0 < port < 65535:
                return utils.make_url(
                    "https" if https else "http",
                    self.manager.host,
                    port,
                    *path,
                    queries=queries)
            return ""

        return lazy_load(make_url)

    def load_nginx_url(
            self: "BaseContainer", key: "ConfigKeyType",
            *path: str, queries: "QueryType | None" = None,
            proxy_name: str = MISSING, proxy_domain_name: str = MISSING,
            proxy_conf: "PathType" = MISSING, proxy_url: str = MISSING,
            https_enable: bool = MISSING, waf_enable: bool = MISSING,
            auth_enable: bool = False, auth_extra: "dict[str, Any]" = None,
    ):

        if not proxy_conf and not proxy_url:
            return ""

        # Snapshot now: auth_extra is caller-owned and may be mutated after
        # this call returns, but the hook key must describe exactly what the
        # hook will write when it fires later, so key and closure share the
        # same frozen copy taken at registration time.
        auth_extra = deepcopy(auth_extra) if auth_extra is not None else None

        def make_url():
            domain = self.get_config(key, type=str, default=None)
            if domain:
                _https = True if https_enable is MISSING else https_enable
                _https = _https and self.get_config("NGINX_HTTPS_ENABLE")
                scheme = "https" if _https else "http"
                port = self.get_config("NGINX_HTTPS_PORT" if _https else "NGINX_HTTP_PORT")
                return utils.make_url(scheme, domain, port, *path, queries=queries)
            return ""

        def make_nginx_conf():
            domain = self.get_config(key, type=str, default=None)
            if domain:

                _https = True if https_enable is MISSING else https_enable
                _https = _https and self.get_config("NGINX_HTTPS_ENABLE")

                _waf = True if waf_enable is MISSING else waf_enable
                _waf = _waf and self.get_config("NGINX_WAF_ENABLE")

                self.write_nginx_conf(
                    domain=domain,
                    proxy_name=proxy_name,
                    proxy_domain_name=proxy_domain_name,
                    proxy_conf=proxy_conf,
                    proxy_url=proxy_url,
                    https_enable=_https,
                    waf_enable=_waf,
                    auth_enable=auth_enable,
                    auth_extra=auth_extra,
                )

        # Include every input that determines the generated conf, so two
        # domains proxying to the same backend each get their own conf
        # written under distinct keys.
        hook_key = (
            "nginx_conf", self._resolve_config_key(key),
            str(proxy_name), str(proxy_domain_name), str(proxy_conf), str(proxy_url),
            str(https_enable), str(waf_enable), auth_enable, _freeze(auth_extra),
        )
        self.add_start_hook(hook_key, make_nginx_conf)
        return lazy_load(make_url)

    def load_exist_nginx_url(self: "BaseContainer", key: "ConfigKeyType",
                             *path: str, queries: "QueryType | None" = None,
                             https: bool = True):
        def make_url():
            nonlocal https
            domain = self.get_config(key, type=str, default=None)
            if domain:
                https = https and self.get_config("NGINX_HTTPS_ENABLE")
                scheme = "https" if https else "http"
                port = self.get_config("NGINX_HTTPS_PORT" if https else "NGINX_HTTP_PORT")
                return utils.make_url(scheme, domain, port, *path, queries=queries)
            return ""

        return lazy_load(make_url)
