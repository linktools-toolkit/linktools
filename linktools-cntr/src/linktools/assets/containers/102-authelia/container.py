#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia container definition."""
import os
import re
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import rsa
import yaml

from linktools import utils
from linktools.cli import CommandError, subcommand
from linktools.cntr import BaseContainer, Flare, Nginx, ContainerError
from linktools.cntr.ext import Authelia, load_nginx_url
from linktools.core import ConfigField, PromptProvider, LazyProvider, AliasProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Mapping
    from linktools.cntr import OperationContext, Integrations
    from linktools.types import PathType


class Container(BaseContainer):
    _config_services = ("authelia", "authelia-admin")
    config_files = ("configuration.yml", "configuration.acl.yml",
                     "configuration.2fa.yml", "configuration.oidc.yml")

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx", "lldap"]

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            AUTHELIA_TAG="latest",
            AUTHELIA_DOMAIN=Nginx.domain(self, "sso"),
            AUTHELIA_LDAP_HOST="lldap",
            AUTHELIA_LDAP_PORT=ConfigField(cast=int, default=3890),
            AUTHELIA_LDAP_ADDRESS=ConfigField(provider=LazyProvider(
                lambda r: f"ldap://{r.get('AUTHELIA_LDAP_HOST')}:{r.get('AUTHELIA_LDAP_PORT')}")),
            AUTHELIA_LDAP_WEB_PORT=ConfigField(cast=int, default=17170),
            AUTHELIA_LDAP_WEB_ADDRESS=ConfigField(provider=LazyProvider(
                lambda r: f"http://{r.get('AUTHELIA_LDAP_HOST')}:{r.get('AUTHELIA_LDAP_WEB_PORT')}")),
            AUTHELIA_LDAP_USER="admin",
            AUTHELIA_LDAP_PASSWORD=ConfigField.chain(
                AliasProvider("LLDAP_ADMIN_PASSWORD"), PromptProvider(cached=True),
            ),
            AUTHELIA_LDAP_BASE_DN=ConfigField.chain(
                AliasProvider("LLDAP_BASE_DN"), default="dc=example,dc=org",
            ),
            AUTHELIA_MIN_AUTH_LEVEL=ConfigField(cast=int, default=2),
            AUTHELIA_OIDC_CLIENT_SECRET=ConfigField(
                provider=LazyProvider(lambda r: utils.random_string(20), cached=True),
            ),
            AUTHELIA_ADMIN_AUTH_ENABLE=ConfigField(cast=bool, default=True),
        )

    @cached_property
    def integrations(self) -> "Integrations":
        return [
            Nginx.site(
                server_name=self.get_config_later("AUTHELIA_DOMAIN"),
                expose=Flare.public(
                    "Authelia", "account", "单点登录", load_nginx_url(self, "web", "auth-admin")),
                template=self.get_source_path("templates", "nginx.conf"),
                auth=None if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else False,
                auth_bypass=(r"\.(css|js)$",),
                auth_rule={"subject": ["group:lldap_admin"]} if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else None,
            ),
        ]

    @cached_property
    def public_url(self) -> str:
        """Use the declared site URL for every externally visible endpoint."""
        # Metadata may describe an unconfigured public identity; native OIDC
        # generation below still requires a concrete HTTPS endpoint.
        return self.manager.nginx_sites[(self.name, "web")].get_url(default="")

    @cached_property
    def public_authority(self) -> str:
        return urlsplit(self.public_url).netloc

    @cached_property
    def public_origin(self) -> str:
        url = urlsplit(self.public_url)
        return urlunsplit((url.scheme, url.netloc, "", "", ""))

    @cached_property
    def _oidc_identity(self) -> "Mapping[str, Any]":
        issuer = self.public_url
        if not issuer.startswith("https://"):
            raise ContainerError("Authelia requires a concrete HTTPS public URL")
        return MappingProxyType({
            "client_id": f"{self.project_name}-web-client",
            "client_name": f"Web Client ({self.project_name})",
            "client_secret": str(self.get_config("AUTHELIA_OIDC_CLIENT_SECRET")),
            "issuer_url": issuer,
            "authorization_url": utils.join_url(issuer, "api/oidc/authorization"),
            "token_url": utils.join_url(issuer, "api/oidc/token"),
            "userinfo_url": utils.join_url(issuer, "api/oidc/userinfo"),
            "user_identifier": "preferred_username",
            "scopes": ("openid", "profile", "groups", "email", "phone"),
        })

    @cached_property
    def oidc_client(self) -> "Mapping[str, Any]":
        """Read-only identity and callbacks rebuilt from the current declarations."""
        client = dict(self._oidc_identity)
        client["redirect_uris"] = self.oidc_redirects
        return MappingProxyType(client)

    @cached_property
    def acl_rules(self) -> "list[dict[str, Any]]":
        rules = []
        for site in self.containers["nginx"].sites.values():
            if not site.enabled or not site.auth or not site.auth_rule:
                continue
            rule = dict(site.auth_rule)
            rule.setdefault(
                "policy",
                "two_factor" if self.get_config("AUTHELIA_MIN_AUTH_LEVEL") > 1 else "one_factor",
            )
            rules.append(rule)
        return rules

    @cached_property
    def oidc_redirects(self) -> "tuple[str, ...]":
        redirects = [self._oidc_identity["issuer_url"]]
        for producer, _, declaration in self.manager.iter_integrations("authelia"):
            if not isinstance(declaration, Authelia):
                raise ContainerError("Invalid Authelia declaration in " + producer.name)
            for redirect in declaration.redirect_uris:
                if redirect not in redirects:
                    redirects.append(redirect)
        return tuple(redirects)

    @cached_property
    def acl_config(self) -> str:
        policy = "two_factor" if self.get_config("AUTHELIA_MIN_AUTH_LEVEL") > 1 else "one_factor"
        rules = list(self.acl_rules)
        root = self.get_config("NGINX_ROOT_DOMAIN")
        rules.append({
            "domain": [root, "*." + root],
            "subject": ["group:admin", "group:admins", "group:super-admin"],
            "policy": policy,
        })
        return yaml.safe_dump(
            {"access_control": {"default_policy": "deny", "rules": rules}},
            sort_keys=False, allow_unicode=True,
        )

    def on_check(self, context: "OperationContext") -> None:
        if context.target_services is not None and not set(context.target_services).intersection(self._config_services):
            return
        if not self.get_config("NGINX_HTTPS_ENABLE"):
            raise ContainerError("Authelia requires HTTPS. Please set NGINX_HTTPS_ENABLE to true.")
        command = ["authelia", "config", "validate"]
        command.extend("--config=/generated/" + name for name in self.config_files)
        result = self.manager.compose_runner.validate_service(context, "authelia", command, check=False)
        if not result.succeeded:
            match = re.search(r" in ([/A-Za-z0-9_.-]+):(\d+)", result.stderr)
            location = " at {}:{}".format(*match.groups()) if match else ""
            raise ContainerError("Native validation failed for authelia{} (exit {})".format(
                location, result.returncode))
    @subcommand("show-notification", help="show notification")
    def on_show_notification(self) -> None:
        path = self.get_app_path("config", "notification.txt")
        if path.exists():
            self.logger.info(utils.read_file(path, text=True))
        else:
            self.logger.warning("No notification.")

    @subcommand("list-oidc-clients", help="list OIDC clients")
    def on_list_oidc_clients(self) -> None:
        self.logger.info(
            yaml.safe_dump(dict(self.oidc_client), sort_keys=False)
        )

    @subcommand("list-acl-rules", help="list acl rules")
    def on_list_acl_rules(self) -> None:
        self.logger.info(
            yaml.dump(self.acl_rules, sort_keys=False)
        )



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


    def on_starting(self, context: "OperationContext") -> None:
        if context.target_services is not None and not set(context.target_services).intersection(self._config_services):
            return
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self.get_app_path("config").mkdir(parents=True, exist_ok=True)
        self.runtime.chmod(secret_path, 0o700, recursive=True)
        for name in ("jwt_secret", "session_secret", "storage_encryption_key", "oidc_hmac_secret"):
            self._create_secret_file(secret_path / name)
        self._create_pem_file(secret_path / "identity_providers_oidc_jwks")
        files = {name: self.render_template(self.get_source_path("templates", name)) for name in self.config_files}
        files["authentication_backend_ldap_password"] = str(self.get_config("AUTHELIA_LDAP_PASSWORD"))
        context.write_files(self, files)