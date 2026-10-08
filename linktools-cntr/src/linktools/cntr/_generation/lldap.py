#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lldap generated configuration and application."""
import os
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cli import CommandError

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext
    from linktools.types import PathType


class LldapGeneration:
    """Own the builtin lldap consumer without extending container hooks."""

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    def prepare(self, context: "EventContext") -> None:
        secret_path = self.container.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self.container.get_app_path("data").mkdir(parents=True, exist_ok=True)
        self.container.runtime.chmod(secret_path, 0o700, recursive=True)
        self._create_secret_file(secret_path / "jwt_secret", length=64)

    def render(self, generation_id: str) -> "dict[str, str]":
        return {
            "lldap_config.toml": self.container.render_template(self.container.get_source_path("templates", "lldap_config.toml")),
            "ldap_user_pass": str(self.container.get_config("LLDAP_ADMIN_PASSWORD")),
        }

    def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        # The builtin TOML contains only fixed database/key locations. LLDAP has
        # no standalone config validator; readiness is checked after application.
        if not self.container.get_config("LLDAP_ADMIN_PASSWORD"):
            raise ContainerError("LLDAP administrator password must not be empty")

    def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
              services: "Iterable[str]") -> None:
        if "lldap" not in services:
            return
        runner = self.container.manager.compose_runner
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
