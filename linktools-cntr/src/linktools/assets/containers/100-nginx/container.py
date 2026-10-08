#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx reverse-proxy container definition."""
import json
import re
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cntr import BaseContainer, ContainerError
from linktools.core import (
    ConfigField, PromptProvider, LazyProvider, AliasProvider, ConfirmProvider,
)
from linktools.decorator import cached_property
from linktools.errors import ConfigNotFoundError
from linktools.rich import choose
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any, Mapping
    from linktools.cntr import EventContext


class Container(BaseContainer):

    @staticmethod
    def _nginx_literal(value: "Any") -> str:
        data = str(value)
        if any(ch in data for ch in ("\r", "\n", "\x00")):
            raise ContainerError("Nginx header value contains a control character")
        data = data.replace("\\", "\\\\").replace('"', '\\"')
        return '"' + data.replace("$", "$" + "{cntr_dollar}") + '"'

    @staticmethod
    def _validated_auth_headers(headers: "dict[str, Any]") -> "dict[str, Any]":
        result = {}
        seen = set()
        forbidden = {
            "host", "forwarded", "x-real-ip", "x-original-url",
            "x-original-method", "x-forwarded-for", "x-forwarded-host",
            "x-forwarded-method", "x-forwarded-proto", "x-forwarded-uri",
            "x-forwarded-port", "x-auth-user", "x-auth-groups",
            "x-auth-name", "x-auth-email",
        }
        for key, value in headers.items():
            if not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_\x60|~0-9A-Za-z-]+", key):
                raise ContainerError("Invalid nginx auth header name")
            normalized = key.lower()
            if normalized in seen or normalized in forbidden or normalized.startswith("x-cntr-"):
                raise ContainerError(f"Duplicate or reserved nginx auth header: {key}")
            seen.add(normalized)
            result[key] = value
        return result

    @cached_property
    def dnsapi(self) -> "dict[str, Any]":
        with open(self.get_source_path("dnsapi.json"), "rt") as fd:
            return json.load(fd)

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            NGINX_TAG="stable-alpine",
            NGINX_WILDCARD_DOMAIN=ConfigField.chain(AliasProvider("WILDCARD_DOMAIN"), default=False),
            NGINX_ROOT_DOMAIN=ConfigField.chain(
                AliasProvider("ROOT_DOMAIN"), PromptProvider(cached=True), default="_",
            ),
            NGINX_HTTP_PORT=ConfigField.chain(
                AliasProvider("HTTP_PORT"), PromptProvider(cached=True), cast=int, default=80,
            ),
            NGINX_HTTPS_ENABLE=ConfigField.chain(
                AliasProvider("HTTPS_ENABLE"), ConfirmProvider(cached=True), cast=bool, default=True,
            ),
            NGINX_HTTPS_PORT=ConfigField(cast=int, default=0, provider=LazyProvider(
                lambda r: 443 if r.get("NGINX_HTTPS_ENABLE") else 0
            )),
            NGINX_DEFAULT_SCHEME=ConfigField(provider=LazyProvider(
                lambda r: "https" if r.get("NGINX_HTTPS_ENABLE") else "http"
            )),
            NGINX_DEFAULT_PORT=ConfigField(provider=LazyProvider(
                lambda r: r.get("NGINX_HTTPS_PORT") if r.get("NGINX_HTTPS_ENABLE") else r.get("NGINX_HTTP_PORT")
            )),
            NGINX_INDEX_URL=ConfigField(provider=LazyProvider(
                lambda r: self._get_default_index_url()
            )),
            NGINX_WAF_ENABLE=ConfigField.chain(
                AliasProvider("WAF_ENABLE"), LazyProvider(lambda r: self.containers["safeline"].enable),
                cast=bool,
            ),
            NGINX_WAF_PORT=ConfigField(cast=int, default=0, provider=LazyProvider(
                lambda r: 8000 if r.get("NGINX_WAF_ENABLE") else 0
            )),
            NGINX_AUTH_ENABLE=ConfigField.chain(
                AliasProvider("AUTH_ENABLE"), LazyProvider(lambda r: self.containers["authelia"].enable),
                cast=bool,
            ),
            ACME_DNS_API=ConfigField.chain(
                LazyProvider(lambda r: self._prompt_acme_dns_api(r), cached=True),
                cast=str, default="",
            ),
        )

    def _prompt_acme_dns_api(self, r):
        # Raise (rather than return "") when HTTPS is disabled, so the
        # enclosing ChainProvider falls through to field.default="" without
        # ever persisting it -- a plain cached=True here would otherwise
        # permanently cache "" the first time this resolves while HTTPS
        # happens to be off, and never prompt again even after HTTPS is
        # enabled later.
        if not r.get("NGINX_HTTPS_ENABLE"):
            raise ConfigNotFoundError("NGINX_HTTPS_ENABLE is disabled")
        return choose("ACME_DNS_API", list(self.dnsapi.keys()))

    @cached_property
    def extend_configs(self) -> "dict[str, Any]":
        configs = {}
        if self.get_config("NGINX_HTTPS_ENABLE"):
            dns_api = self.get_config("ACME_DNS_API")
            if dns_api not in self.dnsapi:
                raise ContainerError(f"Not supported dns_api: {dns_api}")
            env_vars = self.dnsapi.get(dns_api).get("env", {})
            for env_var, meta in env_vars.items():
                configs[env_var] = ConfigField.chain(
                    PromptProvider(cached=True, allow_empty=not meta.get("required", True)),
                    name=env_var, secret=True,
                    default=MISSING if meta.get("required", True) else "",
                )
        return configs

    def _get_default_index_url(self):
        return utils.make_url(
            self.get_config("NGINX_DEFAULT_SCHEME"),
            "www.google.com" \
                if self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost") \
                else self.get_config("NGINX_ROOT_DOMAIN"),
            self.get_config("NGINX_DEFAULT_PORT")
        )

    def on_check(self, context: "EventContext") -> None:
        if self.get_config("NGINX_WILDCARD_DOMAIN") and self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost"):
            raise ContainerError("Wildcard domain is enabled but root domain is not set.")
        if self.get_config("NGINX_WAF_ENABLE") and not self.containers["safeline"].enable:
            raise ContainerError("NGINX_WAF_ENABLE is true but safeline container is not enabled.")
        if self.get_config("NGINX_AUTH_ENABLE") and not self.containers["authelia"].enable:
            raise ContainerError("NGINX_AUTH_ENABLE is true but authelia container is not enabled.")

    def quote(self, value: "Any") -> str:
        return self._nginx_literal(value)

    @cached_property
    def sites(self) -> "Mapping[tuple[str, str], Any]":
        return self.manager.nginx_sites

    def complex_value(self, value: str) -> str:
        """Quote one native nginx complex value without interpreting its variables."""
        if not isinstance(value, str) or any(ch in value for ch in ("\r", "\n", "\x00")):
            raise ContainerError("Invalid nginx complex value")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def header_items(self, site: "Any", overrides: "Mapping[str, str] | None" = None,
                     authentication: bool = False) -> "tuple[tuple[str, str], ...]":
        """Build a complete case-insensitive header set for one proxy location."""
        headers = {
            "Host": "$cntr_host", "Upgrade": "$http_upgrade",
            "Connection": "$connection_upgrade", "X-Real-IP": "$cntr_client_ip",
            "X-Original-URL": "$cntr_scheme://$cntr_host$cntr_uri",
            "X-Original-Method": "$cntr_method", "X-Forwarded-Proto": "$cntr_scheme",
            "X-Forwarded-Host": "$cntr_host", "X-Forwarded-URI": "$cntr_uri",
            "X-Forwarded-Method": "$cntr_method", "X-Forwarded-For": "$cntr_client_ip",
            "X-Forwarded-Port": "$cntr_port", "Forwarded": "",
        }
        for name in ("Scheme", "Host", "URI", "Method", "Client-IP"):
            headers["X-Cntr-" + name] = ""
        for name in ("User", "Groups", "Name", "Email"):
            headers["X-Auth-" + name] = (
                "$cntr_identity_" + site.var_name + "_" + name.lower()
                if site.auth and not authentication else "")
        auth_headers = self._validated_auth_headers(site.auth_headers) if site.auth else {}
        if not authentication:
            for index, key in enumerate(auth_headers):
                headers[key] = "$cntr_credential_" + site.var_name + "_" + str(index)
        seen = set()
        protected = {key.lower() for key in auth_headers}
        for key, value in (overrides or {}).items():
            if not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key):
                raise ContainerError("Invalid nginx header override name")
            normalized = key.lower()
            if (normalized in seen or normalized.startswith("x-cntr-") or
                    normalized in {"x-auth-user", "x-auth-groups", "x-auth-name", "x-auth-email"} or
                    (not authentication and normalized in protected)):
                raise ContainerError("Duplicate or reserved nginx header override: " + key)
            seen.add(normalized)
            self.complex_value(value)
            for original in tuple(headers):
                if original.lower() == normalized:
                    del headers[original]
            headers[key] = value
        if authentication:
            headers.update({"Content-Length": "", "Connection": "", "Upgrade": ""})
        return tuple((key, self.complex_value(value)) for key, value in headers.items())

    def security_maps(self, site: "Any") -> str:
        """Generate only maps whose inputs and evaluation phase are explicit."""
        lines = []
        for capability in ("waf", "auth"):
            if not getattr(site, capability):
                continue
            lines.extend(["map $uri $cntr_" + capability + "_skip_" + site.var_name + " {", "    default 0;"])
            for regex in getattr(site, capability + "_bypass"):
                lines.append("    " + self.complex_value("~*" + regex) + " 1;")
            lines.append("}")
        if not site.auth:
            return "\n".join(lines)
        suffix = site.var_name
        # Exact keys preserve business regex captures when evaluated at proxy time.
        lines.append('map "$cntr_auth_status_' + suffix + ':$cntr_auth_skip_' + suffix + ':$cntr_auth_proof_' + suffix + '" $cntr_authorized_' + suffix + ' {')
        lines.append("    volatile;")
        lines.append("    default 0;")
        lines.extend('    "' + str(status) + ':0:1" 1;' for status in range(200, 300))
        lines.append("}")
        for name in ("user", "groups", "name", "email"):
            lines.extend([
                "map $cntr_authorized_" + suffix + " $cntr_identity_" + suffix + "_" + name + " {",
                '    volatile;', '    default "";', "    1 $cntr_auth_" + name + "_" + suffix + ";", "}",
            ])
        for index, (key, value) in enumerate(self._validated_auth_headers(site.auth_headers).items()):
            # nginx ignores non-conventional incoming header names by default;
            # its $http_* variable syntax cannot address their punctuation.
            incoming = ("${http_" + key.lower().replace("-", "_") + "}"
                        if re.fullmatch(r"[A-Za-z0-9_-]+", key) else '""')
            lines.extend([
                "map $cntr_authorized_" + suffix + " $cntr_credential_" + suffix + "_" + str(index) + " {",
                "    volatile;", "    default " + incoming + ";", "    1 " + self.quote(value) + ";", "}",
            ])
        return "\n".join(lines)
