#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portainer container definition."""
from typing import TYPE_CHECKING

from linktools.core import ConfigField
from linktools.decorator import cached_property
from linktools.runtime import lazy_load
from linktools.cntr import BaseContainer, NginxSite

if TYPE_CHECKING:
    from typing import Any


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            PORTAINER_TAG="alpine",
            PORTAINER_DOMAIN=self.get_nginx_domain(),
            PORTAINER_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            PORTAINER_PORT=ConfigField(cast=int, default=9000),
        )

    @cached_property
    def integrations(self) -> "dict[str, dict[str, Any]]":
        return {
            "flare": {
                "web": self.expose_public("Portainer", "docker", "Docker管理工具",
                                         self.load_nginx_url("web")),
                "direct": self.expose_container("Portainer", "docker", "Docker管理工具", self.load_port_url(
                    "PORTAINER_PORT",
                    https=False
                )),
            },
            "nginx": {
                "web": NginxSite(
                    server_name=self.get_config_later("PORTAINER_DOMAIN"),
                    proxy="http://portainer:9000",
                    auth=None if self.get_config("PORTAINER_AUTH_ENABLE") else False,
                    auth_bypass=(r"\.(css|js)$",),
                    oidc_redirects=lazy_load(lambda: ("",) if (
                        self.get_config("PORTAINER_AUTH_ENABLE")
                        and self.get_config("NGINX_AUTH_ENABLE")) else ()),
                ),
            },
        }
