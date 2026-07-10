#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Access-link declarations (exposes) for a container: categories, lazily
resolved URLs, and the idempotent nginx-conf registration hook."""
from typing import TYPE_CHECKING

from linktools import utils
from linktools.runtime import lazy_load
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import ConfigKeyType, PathType, QueryType
    from ..container import BaseContainer


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

        # Idempotent: a container's nginx conf for a given proxy_conf/proxy_url is
        # written once even if load_nginx_url is re-evaluated.
        self.add_start_hook(("nginx_conf", str(proxy_conf), str(proxy_url)), make_nginx_conf)
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
