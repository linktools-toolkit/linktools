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
    from typing import Any
    from pathlib import Path
    from linktools.cntr.artifacts import GeneratedCandidate
    from collections.abc import Iterable
    from linktools.cntr import EventContext, ExposeLink


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        def get_base_dn(cfg: "dict[str, Any]") -> str:
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

    def on_check(self, context: "EventContext") -> None:
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if not domain or "." not in domain:
            raise ContainerError(f"Invalid domain `{domain}` for LDAP, "
                                 f"Please set NGINX_ROOT_DOMAIN to a valid domain (e.g., example.com).")

    @property
    def generated_config_path(self) -> "Path":
        return self.get_app_path("generated")

    def prepare_generated_config(self, context: "EventContext") -> None:
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self.get_app_path("data").mkdir(parents=True, exist_ok=True)
        self.runtime.chmod(secret_path, 0o700, recursive=True)
        self._create_secret_file(secret_path / "jwt_secret", length=64)

    def render_generated_config(self, generation_id: str) -> "dict[str, str]":
        return {
            "lldap_config.toml": self.render_template(self.get_source_path("templates", "lldap_config.toml")),
            "ldap_user_pass": str(self.get_config("LLDAP_ADMIN_PASSWORD")),
        }

    def validate_generated_config(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        # The builtin TOML contains only fixed database/key locations. LLDAP has
        # no standalone config validator; readiness is checked after application.
        if not self.get_config("LLDAP_ADMIN_PASSWORD"):
            raise ContainerError("LLDAP administrator password must not be empty")

    def apply_generated_config(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        runner = self.manager.compose_runner
        recreate = candidate.changed or not runner.is_generation_current(context, "lldap", candidate)
        runner.apply_service(context, "lldap", recreate=recreate)
        runner.wait_service_healthy(context, "lldap")

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))
