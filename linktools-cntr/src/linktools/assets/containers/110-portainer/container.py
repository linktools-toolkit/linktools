#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portainer container definition."""
from typing import TYPE_CHECKING

from linktools.core import ConfigField
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from linktools.cntr import ExposeLink


class Container(BaseContainer):

    @cached_property
    def configs(self):
        return dict(
            PORTAINER_TAG="alpine",
            PORTAINER_DOMAIN=self.get_nginx_domain(),
            PORTAINER_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            PORTAINER_PORT=ConfigField(cast=int, default=9000),
        )

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_public("Portainer", "docker", "Docker管理工具", self.load_nginx_url(
                "PORTAINER_DOMAIN",
                proxy_url="http://portainer:9000",
                auth_enable=self.get_config("PORTAINER_AUTH_ENABLE"),
                auth_extra={
                    "acl_bypass": ["\\.(css|js)$"],
                    "oidc_redirect_uris": ["{base_url}"]
                }
            )),
            self.expose_container("Portainer", "docker", "Docker管理工具", self.load_port_url(
                "PORTAINER_PORT",
                https=False
            )),
        ]
