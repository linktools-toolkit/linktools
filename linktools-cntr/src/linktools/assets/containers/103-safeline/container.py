#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SafeLine WAF container definition."""
from typing import TYPE_CHECKING

from linktools.cli import subcommand
from linktools.cntr import BaseContainer
from linktools.core import ConfigField
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from collections.abc import Iterable
    from linktools.cntr import ExposeLink


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx"]

    @cached_property
    def configs(self):
        return dict(
            SAFELINE_TAG="latest",
            SAFELINE_IMAGE_PREFIX="chaitin",
            SAFELINE_DOMAIN=self.get_nginx_domain(),
            SAFELINE_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            SAFELINE_POSTGRES_PASSWORD="Pg-pAssw0rd",
            SAFELINE_SUBNET_PREFIX="172.22.242",
            SAFELINE_ARCH_SUFFIX="",
            SAFELINE_RELEASE="",
            SAFELINE_PORT=ConfigField(cast=int, default=9200),
            SAFELINE_API_TOKEN="",
        )

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_public("Safeline", "alienOutline", "雷池WAF", self.load_nginx_url(
                "SAFELINE_DOMAIN",
                proxy_url="https://safeline-mgt:1443",
                auth_enable=self.get_config("SAFELINE_AUTH_ENABLE"),
                auth_extra={
                    "acl_bypass": ["\\.(css|js)$"],
                    "auth_headers": {
                        "X-SLCE-API-TOKEN": self.get_config("SAFELINE_API_TOKEN"),
                    }
                },
            )),
            self.expose_container("Safeline", "alienOutline", "雷池WAF", self.load_port_url(
                "SAFELINE_PORT",
                https=True
            )),
        ]

    @subcommand("reset-admin", help="reset safeline admin password")
    def on_reset_admin(self):
        self.runtime.create_docker_process(
            "exec", "-it", self.get_service_name("safeline-mgt"),
            "resetadmin"
        ).call()
