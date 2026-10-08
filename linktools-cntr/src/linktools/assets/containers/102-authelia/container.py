#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia container definition."""

import os
from typing import TYPE_CHECKING, Mapping, Any

import rsa
import yaml

from linktools import utils
from linktools.cli import CommandError, subcommand
from linktools.cntr import BaseContainer, NginxSite, ContainerError
from linktools.core import ConfigField, PromptProvider, LazyProvider, AliasProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable
    from linktools.cntr import EventContext, ExposeLink


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx", "lldap"]

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            AUTHELIA_TAG="latest",
            AUTHELIA_DOMAIN=self.get_nginx_domain("sso"),
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
    def integrations(self) -> "dict[str, dict[str, NginxSite]]":
        return {
            "nginx": {
                "web": NginxSite(
                    server_name=self.get_config_later("AUTHELIA_DOMAIN"),
                    template=self.get_source_path("templates", "nginx.conf"),
                    auth=None if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else False,
                    auth_bypass=(r"\.(css|js)$",),
                    auth_rule={"subject": ["group:lldap_admin"]} if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else None,
                ),
            },
        }

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_public("Authelia", "account", "单点登录", self.load_nginx_url("web")),
        ]

    @cached_property
    def oidc_client(self) -> "Mapping[str, Any]":
        """Read-only OIDC connection details, independent of site redirects."""
        from types import MappingProxyType

        domain = self.get_config("AUTHELIA_DOMAIN")
        issuer = utils.make_url("https", domain, self.get_config("NGINX_HTTPS_PORT"))
        return MappingProxyType({
            "client_id": f"{self.project_name}-web-client",
            "client_name": f"Web Client ({self.project_name})",
            "client_secret": self.get_config("AUTHELIA_OIDC_CLIENT_SECRET"),
            "issuer_url": issuer,
            "authorization_url": issuer + "/api/oidc/authorization",
            "token_url": issuer + "/api/oidc/token",
            "userinfo_url": issuer + "/api/oidc/userinfo",
            "user_identifier": "preferred_username",
            "scopes": ("openid", "profile", "groups", "email", "phone"),
        })

    @cached_property
    def acl_rules(self) -> "list[dict[str, Any]]":
        """Regenerate native access rules from this project's current sites."""
        from linktools.cntr import NginxSite

        rules = []
        for producer, site_id, site in self.manager.iter_integrations("nginx"):
            if not isinstance(site, NginxSite):
                raise ContainerError(f"Invalid nginx integration {producer.name}/{site_id}")
            domain = str(site.server_name)
            if not domain or not site.auth_rule:
                continue
            rule = dict(site.auth_rule)
            if "domain" not in rule and "domain_regex" not in rule:
                if domain.startswith("~") or "*" in domain or " " in domain or domain == "_":
                    raise ContainerError(
                        f"Authelia rule for {producer.name}/{site_id} requires a native domain")
                rule["domain"] = [domain]
            rule.setdefault(
                "policy",
                "two_factor" if self.get_config("AUTHELIA_MIN_AUTH_LEVEL") > 1 else "one_factor",
            )
            rules.append(rule)
        return rules

    @cached_property
    def oidc_redirects(self) -> "tuple[str, ...]":
        """Rebuild currently declared callbacks, never restoring stale derived state."""
        from urllib.parse import urlsplit, urlunsplit
        from linktools.cntr import NginxSite

        redirects = [self.oidc_client["issuer_url"]]
        for producer, site_id, site in self.manager.iter_integrations("nginx"):
            if not isinstance(site, NginxSite):
                raise ContainerError(f"Invalid nginx integration {producer.name}/{site_id}")
            domain = str(site.server_name)
            if not domain or not site.oidc_redirects:
                continue
            if domain == "_" or domain.startswith("~") or "*" in domain or " " in domain:
                if not site.url:
                    raise ContainerError(f"OIDC site {producer.name}/{site_id} requires a URL")
            base = str(site.url) if site.url else utils.make_url(
                "https" if site.https is not False else "http", domain,
                self.get_config("NGINX_HTTPS_PORT" if site.https is not False else "NGINX_HTTP_PORT")
            )
            if "{{" in base or "}}" in base:
                raise ContainerError(f"OIDC site {producer.name}/{site_id} has a template URL")
            for redirect in site.oidc_redirects:
                target = str(redirect)
                if target.startswith("//"):
                    raise ContainerError(f"OIDC site {producer.name}/{site_id} has a protocol-relative URI")
                if not target:
                    target = base
                elif target.startswith("/"):
                    parsed = urlsplit(base)
                    target = urlunsplit((parsed.scheme, parsed.netloc, target, "", ""))
                parsed = urlsplit(target)
                if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
                    raise ContainerError(f"OIDC site {producer.name}/{site_id} has an invalid redirect URI")
                if target not in redirects:
                    redirects.append(target)
        return tuple(redirects)

    @cached_property
    def oidc_clients(self) -> "list[dict[str, Any]]":
        """Native Authelia authoring structure derived from one read-only client."""
        client = self.oidc_client
        return [{
            "ClientID": client["client_id"],
            "ClientName": client["client_name"],
            "ClientSecret": client["client_secret"],
            "IssuerURL": client["issuer_url"],
            "AuthorizationURL": client["authorization_url"],
            "AccessTokenURL": client["token_url"],
            "ResourceURL": client["userinfo_url"],
            "RedirectURLs": self.oidc_redirects,
            "UserIdentifier": client["user_identifier"],
            "Scopes": " ".join(client["scopes"]),
        }]

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

    def on_init(self) -> None:
        self.start_hooks.append(lambda: self.manager.start_hooks.append(self._update_files))

    def on_check(self, context: "EventContext") -> None:
        if not self.get_config("NGINX_HTTPS_ENABLE"):
            raise ContainerError("Authelia requires HTTPS. Please set NGINX_HTTPS_ENABLE to true.")

    def _update_files(self):
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        config_path = self.get_app_path("config")
        config_path.mkdir(parents=True, exist_ok=True)
        template_path = self.get_source_path("templates")

        self.runtime.chown(secret_path, self.user, recursive=True)
        self.runtime.chmod(secret_path, 0o700, recursive=True)
        self.runtime.chown(config_path, self.user, recursive=True)
        self.runtime.chmod(config_path, 0o700, recursive=True)

        self._create_secret_file(secret_path / "jwt_secret")
        self._create_secret_file(secret_path / "session_secret")
        self._create_secret_file(secret_path / "storage_encryption_key")
        self._create_secret_file(secret_path / "oidc_hmac_secret")
        self._create_pem_file(secret_path / "identity_providers_oidc_jwks")
        utils.write_file(secret_path / "authentication_backend_ldap_password", self.get_config("LLDAP_ADMIN_PASSWORD"))

        self.render_template(template_path / "configuration.yml", config_path / "configuration.yml")
        self.render_template(template_path / "configuration.acl.yml", config_path / "configuration.acl.yml")
        self.render_template(template_path / "configuration.2fa.yml", config_path / "configuration.2fa.yml")
        self.render_template(template_path / "configuration.oidc.yml", config_path / "configuration.oidc.yml")

        self.runtime.chown(secret_path, "root", recursive=True)
        self.runtime.chown(config_path, "root", recursive=True)

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
            yaml.dump(self.oidc_clients, sort_keys=False)
        )

    @subcommand("list-acl-rules", help="list acl rules")
    def on_list_acl_rules(self) -> None:
        self.logger.info(
            yaml.dump(self.acl_rules, sort_keys=False)
        )

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))

    @classmethod
    def _create_pem_file(cls, path):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        public_key, private_key = rsa.newkeys(nbits=2048, exponent=65537)
        private_pem = private_key.save_pkcs1(format="PEM")
        utils.write_file(path, private_pem)
