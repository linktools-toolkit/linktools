#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authelia container definition."""

from typing import TYPE_CHECKING
from types import MappingProxyType

import yaml

from linktools import utils
from linktools.cli import subcommand
from linktools.cntr import BaseContainer, ExposeLink, NginxSite, ContainerError
from linktools.cntr.urls import load_nginx_url
from linktools.core import ConfigField, PromptProvider, LazyProvider, AliasProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from linktools.cntr import Integrations
    from typing import Any, Mapping
    from collections.abc import Iterable
    from linktools.cntr import EventContext


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
    def integrations(self) -> "Integrations":
        return {
            "nginx": {
                "web": NginxSite(
                    expose=ExposeLink.public(
                        "Authelia", "account", "单点登录", load_nginx_url(self, "web", "auth-admin")),
                    server_name=self.get_config_later("AUTHELIA_DOMAIN"),
                    template=self.get_source_path("templates", "nginx.conf"),
                    auth=None if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else False,
                    auth_bypass=(r"\.(css|js)$",),
                    auth_rule={"subject": ["group:lldap_admin"]} if self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE") else None,
                ),
            },
        }

    @cached_property
    def _oidc_identity(self) -> "Mapping[str, Any]":
        issuer = str(load_nginx_url(self, "web"))
        if not issuer.startswith("https://"):
            raise ContainerError("Authelia requires a concrete HTTPS public URL")
        return MappingProxyType({
            "client_id": f"{self.project_name}-web-client",
            "client_name": f"Web Client ({self.project_name})",
            "client_secret": str(self.get_config("AUTHELIA_OIDC_CLIENT_SECRET")),
            "issuer_url": issuer,
            "authorization_url": issuer + "/api/oidc/authorization",
            "token_url": issuer + "/api/oidc/token",
            "userinfo_url": issuer + "/api/oidc/userinfo",
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
        for site in self.containers["nginx"].sites.values():
            if site.enabled:
                for redirect in site.oidc_redirects:
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

    def on_check(self, context: "EventContext") -> None:
        if not self.get_config("NGINX_HTTPS_ENABLE"):
            raise ContainerError("Authelia requires HTTPS. Please set NGINX_HTTPS_ENABLE to true.")

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
