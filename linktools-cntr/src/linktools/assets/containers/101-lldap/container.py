#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLDAP container definition."""
import os
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cli import CommandError
from linktools.cntr import BaseContainer, ContainerError
from linktools.core import ConfigField, PromptProvider, LazyProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from collections.abc import Iterable
    from linktools.cntr import EventContext, ExposeLink


class Container(BaseContainer):

    @cached_property
    def configs(self):
        def get_base_dn(cfg):
            domain = cfg.get("NGINX_ROOT_DOMAIN")
            parts = domain.split(".")
            return ",".join([f"dc={part}" for part in parts])

        return dict(
            LLDAP_TAG="stable",
            LLDAP_DOMAIN=self.get_nginx_domain("ldap"),
            LLDAP_PORT=ConfigField(cast=int, default=0),
            LLDAP_WEB_PORT=ConfigField(cast=int, default=0),
            LLDAP_BASE_DN=ConfigField(provider=LazyProvider(lambda r: get_base_dn(r))),
            LLDAP_ADMIN_PASSWORD=ConfigField(provider=PromptProvider(
                default=utils.random_string(20), cached=True,
            )),
        )

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_container("LDAP", "account", "账号管理", self.load_port_url(
                "LLDAP_WEB_PORT",
                https=False,
            )),
        ]

    def on_check(self, context: "EventContext"):
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if not domain or "." not in domain:
            raise ContainerError(f"Invalid domain `{domain}` for LDAP, "
                                 f"Please set NGINX_ROOT_DOMAIN to a valid domain (e.g., example.com).")

    def on_starting(self, context: "EventContext"):
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)

        data_path = self.get_app_path("data")
        data_path.mkdir(parents=True, exist_ok=True)

        template_path = self.get_source_path("templates")

        self.runtime.chown(secret_path, self.user, recursive=True)
        self.runtime.chmod(secret_path, 0o700, recursive=True)
        self.runtime.chown(data_path, self.user, recursive=True)
        self.runtime.chmod(data_path, 0o700, recursive=True)

        self._create_secret_file(secret_path / "jwt_secret", length=64)
        utils.write_file(secret_path / "ldap_user_pass", self.get_config("LLDAP_ADMIN_PASSWORD"))
        self.render_template(template_path / "lldap_config.toml", data_path / "lldap_config.toml")

        self.runtime.chown(secret_path, "root", recursive=True)
        self.runtime.chown(data_path, "root", recursive=True)

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))
