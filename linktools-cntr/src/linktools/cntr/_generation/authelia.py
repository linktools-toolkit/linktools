#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia generated configuration and application."""
import os
from typing import TYPE_CHECKING

import rsa

from linktools import utils
from linktools.cli import CommandError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext
    from linktools.types import PathType


class AutheliaGeneration:
    """Own the builtin authelia consumer without extending container hooks."""

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    def prepare(self, context: "EventContext") -> None:
        secret_path = self.container.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self.container.get_app_path("config").mkdir(parents=True, exist_ok=True)
        self.container.runtime.chmod(secret_path, 0o700, recursive=True)
        for name in ("jwt_secret", "session_secret", "storage_encryption_key", "oidc_hmac_secret"):
            self._create_secret_file(secret_path / name)
        self._create_pem_file(secret_path / "identity_providers_oidc_jwks")

    def render(self, generation_id: str) -> "dict[str, str]":
        result = {
            name: self.container.render_template(self.container.get_source_path("templates", name))
            for name in ("configuration.yml", "configuration.acl.yml",
                         "configuration.2fa.yml", "configuration.oidc.yml")
        }
        result["authentication_backend_ldap_password"] = str(self.container.get_config("AUTHELIA_LDAP_PASSWORD"))
        return result

    def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        root = "/generated/" + candidate.generation_id
        command = ["authelia", "config", "validate"]
        command.extend("--config=" + root + "/" + name for name in (
            "configuration.yml", "configuration.acl.yml",
            "configuration.2fa.yml", "configuration.oidc.yml"))
        self.container.manager.compose_runner.validate_service(
            context, "authelia", command,
            environment={"AUTHELIA_AUTHENTICATION_BACKEND_LDAP_PASSWORD_FILE":
                         root + "/authentication_backend_ldap_password"},
        )

    def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
              services: "Iterable[str]") -> None:
        runner = self.container.manager.compose_runner
        services = tuple(services)
        for service in services:
            if service not in ("authelia", "authelia-admin"):
                runner.apply_service(context, service)
        if "authelia" in services:
            recreate = candidate.changed or not runner.is_generation_current(context, "authelia", candidate)
            runner.apply_service(context, "authelia", recreate=recreate)
            runner.wait_service_healthy(context, "authelia")
        if "authelia-admin" in services:
            base_changed = "configuration.yml" in candidate.changed_files
            runner.apply_service(context, "authelia-admin", recreate=base_changed)

    @classmethod
    def _create_secret_file(cls, path: "PathType", length: int = 48) -> None:
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))

    @classmethod
    def _create_pem_file(cls, path: "PathType") -> None:
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        public_key, private_key = rsa.newkeys(nbits=2048, exponent=65537)
        private_pem = private_key.save_pkcs1(format="PEM")
        utils.write_file(path, private_pem)
