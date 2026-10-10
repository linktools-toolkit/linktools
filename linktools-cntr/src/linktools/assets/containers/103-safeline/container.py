#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SafeLine WAF container definition."""
from typing import TYPE_CHECKING

from linktools.cli import subcommand
from linktools.cntr import BaseContainer, Flare, Nginx
from linktools.cntr.ext import load_port_url
from linktools.core import ConfigField
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any
    from linktools.cntr import OperationContext, Integrations


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx"]

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            SAFELINE_TAG="latest",
            SAFELINE_IMAGE_PREFIX="chaitin",
            SAFELINE_DOMAIN=Nginx.domain(self),
            SAFELINE_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            SAFELINE_POSTGRES_PASSWORD="Pg-pAssw0rd",
            SAFELINE_SUBNET_PREFIX="172.22.242",
            SAFELINE_ARCH_SUFFIX="",
            SAFELINE_REGION="",
            SAFELINE_PORT=ConfigField(cast=int, default=9200),
            SAFELINE_API_TOKEN="",
        )

    @cached_property
    def integrations(self) -> "Integrations":
        return [
            Flare.container("Safeline", "alienOutline", load_port_url(
                self, "SAFELINE_PORT",
                https=True
            )),
            Nginx.site(
                self.get_config_later("SAFELINE_DOMAIN"),
                proxy="https://safeline-mgt:1443",
                auth=None if self.get_config("SAFELINE_AUTH_ENABLE") else False,
                auth_bypass=(r"\.(css|js)$",),
                auth_headers={"X-SLCE-API-TOKEN": self.get_config_later("SAFELINE_API_TOKEN")},
                link=Flare.public("Safeline", "alienOutline", "雷池WAF"),
            ),
        ]

    @subcommand("reset-admin", help="reset safeline admin password")
    def on_reset_admin(self) -> None:
        self.runtime.create_docker_process(
            "exec", "-it", self.get_service_name("safeline-mgt"),
            "resetadmin"
        ).call()

