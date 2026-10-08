#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLDAP container definition."""
import os
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cli import CommandError
from linktools.cntr import BaseContainer, Flare, Nginx, ContainerError
from linktools.cntr.integration import load_port_url
from linktools.core import ConfigField, PromptProvider, LazyProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any
    from linktools.cntr import EventContext, Integrations
    from linktools.cntr.artifacts import GeneratedCandidate
    from linktools.types import PathType
    from linktools.core import ConfigResolver


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        def get_base_dn(cfg: "ConfigResolver") -> str:
            domain = cfg.get("NGINX_ROOT_DOMAIN")
            parts = domain.split(".")
            return ",".join([f"dc={part}" for part in parts])

        return dict(
            LLDAP_TAG="stable",
            LLDAP_DOMAIN=Nginx.domain(self, "ldap"),
            LLDAP_PORT=ConfigField(cast=int, default=0),
            LLDAP_WEB_PORT=ConfigField(cast=int, default=0),
            LLDAP_BASE_DN=ConfigField(provider=LazyProvider(lambda r: get_base_dn(r))),
            LLDAP_ADMIN_PASSWORD=ConfigField(provider=PromptProvider(
                default=utils.random_string(20), cached=True,
            )),
        )

    @cached_property
    def integrations(self) -> "Integrations":
        return [
            Flare.bookmark("LDAP", "account", load_port_url(
                self, "LLDAP_WEB_PORT",
                https=False,
            ), category="container"),
        ]

    def on_check(self, context: "EventContext") -> None:
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if not domain or "." not in domain:
            raise ContainerError(f"Invalid domain `{domain}` for LDAP, "
                                 f"Please set NGINX_ROOT_DOMAIN to a valid domain (e.g., example.com).")


    generates_config = True

    def on_prepare_config(self, context: "EventContext") -> None:
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self.get_app_path("data").mkdir(parents=True, exist_ok=True)
        self.runtime.chmod(secret_path, 0o700, recursive=True)
        self._create_secret_file(secret_path / "jwt_secret", length=64)

    def render_config(self, generation_id: str) -> "dict[str, str]":
        return {
            "lldap_config.toml": self.render_template(self.get_source_path("templates", "lldap_config.toml")),
            "ldap_user_pass": str(self.get_config("LLDAP_ADMIN_PASSWORD")),
        }

    def validate_config(self, context: "EventContext", candidate: "GeneratedCandidate") -> None:
        # The builtin TOML contains only fixed database/key locations. LLDAP has
        # no standalone config validator; readiness is checked after application.
        if not self.get_config("LLDAP_ADMIN_PASSWORD"):
            raise ContainerError("LLDAP administrator password must not be empty")

    def apply_config(self, context: "EventContext", candidate: "GeneratedCandidate",
                 services: "Iterable[str]") -> None:
        if "lldap" not in services:
            return
        runner = self.manager.compose_runner
        recreate = candidate.changed or not runner.is_generation_current(context, "lldap", candidate)
        runner.apply_service(context, "lldap", recreate=recreate)
        runner.wait_service_healthy(context, "lldap")

    @classmethod
    def _create_secret_file(cls, path: "PathType", length: int = 48) -> None:
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))
