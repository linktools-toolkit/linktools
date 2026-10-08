#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx reverse-proxy container definition."""
import json
import os
import re
from copy import copy
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cntr import BaseContainer, ContainerError, NginxSite
from linktools.core import (
    ConfigField, PromptProvider, LazyProvider, AliasProvider, ConfirmProvider,
)
from linktools.decorator import cached_property
from linktools.errors import ConfigNotFoundError
from linktools.rich import choose
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
    from linktools.cntr import EventContext
    from linktools.types import PathType


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
                    PromptProvider(cached=True, allow_empty=meta.get("required", True)),
                    name=env_var,
                )
        return configs

    @cached_property
    def _acme_ssl_domains(self):
        result = []
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            result.extend([domain, f"*.{domain}"])
        if self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            for producer, local_id, site in self.manager.iter_integrations("nginx"):
                if not isinstance(site, NginxSite):
                    raise ContainerError(
                        f"Invalid nginx site {producer.name}/{local_id}")
                if not str(site.server_name) or site.https is False:
                    continue
                for cert_domain in site.cert_domains:
                    value = str(cert_domain)
                    if value and value not in result:
                        result.append(value)
        return result

    @cached_property
    def acme_ssl_domains_args(self) -> str:
        return " ".join([f"--domain {domain}" for domain in self._acme_ssl_domains if domain])

    @cached_property
    def acme_ssl_certificate_args(self) -> str:
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            return " ".join([
                "--cert-file", f"/etc/certs/{domain}_cert.pem",
                "--key-file", f"/etc/certs/{domain}_key.pem",
                "--fullchain-file", f"/etc/certs/{domain}_fullchain.pem",
            ])
        return ""

    def append_ssl_domains(self, *domians: str) -> None:
        for domain in domians:
            if domain and domain not in self._acme_ssl_domains:
                self._acme_ssl_domains.append(domain)

    def _get_default_index_url(self):
        return utils.make_url(
            self.get_config("NGINX_DEFAULT_SCHEME"),
            "www.google.com" \
                if self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost") \
                else self.get_config("NGINX_ROOT_DOMAIN"),
            self.get_config("NGINX_DEFAULT_PORT")
        )

    def on_init(self) -> None:
        self.start_hooks.append(lambda: self.manager.start_hooks.append(self._update_files))

    def on_check(self, context: "EventContext") -> None:
        if self.get_config("NGINX_WILDCARD_DOMAIN") and self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost"):
            raise ContainerError("Wildcard domain is enabled but root domain is not set.")
        if self.get_config("NGINX_WAF_ENABLE") and not self.containers["safeline"].enable:
            raise ContainerError("NGINX_WAF_ENABLE is true but safeline container is not enabled.")
        if self.get_config("NGINX_AUTH_ENABLE") and not self.containers["authelia"].enable:
            raise ContainerError("NGINX_AUTH_ENABLE is true but authelia container is not enabled.")

    def quote(self, value: "Any") -> str:
        return self._nginx_literal(value)

    def _write_site(self, producer: "BaseContainer", local_id: str, declaration: "NginxSite") -> None:
        domain = str(declaration.server_name)
        if not domain:
            return
        if declaration.proxy is None and declaration.template is None:
            raise ContainerError(f"Nginx site {producer.name}/{local_id} has no proxy or template")
        for capability, key in (
                ("https", "NGINX_HTTPS_ENABLE"),
                ("waf", "NGINX_WAF_ENABLE"),
                ("auth", "NGINX_AUTH_ENABLE")):
            if getattr(declaration, capability) is True and not self.get_config(key, type=bool):
                raise ContainerError(
                    f"Nginx site {producer.name}/{local_id} requires disabled {capability}")
        https = self.get_config("NGINX_HTTPS_ENABLE", type=bool) if declaration.https is None else declaration.https
        waf = self.get_config("NGINX_WAF_ENABLE", type=bool) if declaration.waf is None else declaration.waf
        auth = self.get_config("NGINX_AUTH_ENABLE", type=bool) if declaration.auth is None else declaration.auth
        if auth and not https:
            raise ContainerError(f"Nginx site {producer.name}/{local_id} requires HTTPS")
        if auth and not self.containers["authelia"].enable:
            raise ContainerError(f"Nginx site {producer.name}/{local_id} requires Authelia")
        if waf and not self.containers["safeline"].enable:
            raise ContainerError(f"Nginx site {producer.name}/{local_id} requires SafeLine")

        file_id = "site_" + producer.name.encode("utf-8").hex() + "_" + local_id.encode("utf-8").hex()
        site = copy(declaration)
        site.server_name = domain
        site.https, site.waf, site.auth = https, waf, auth
        site.waf_bypass = tuple(str(value) for value in declaration.waf_bypass) if waf else ()
        site.auth_bypass = tuple(str(value) for value in declaration.auth_bypass) if auth else ()
        site.auth_headers = self._validated_auth_headers(declaration.auth_headers) if auth else {}
        site.file_id = file_id
        site.var_name = file_id.encode("utf-8").hex()

        server_path = self.get_app_path("conf.d", file_id + ".conf", create_parent=True)
        fragment_path = self.get_app_path("conf.d", file_id + "_confs", file_id + ".conf", create_parent=True)
        source = declaration.template or self.get_source_path("templates", "default.conf")
        utils.write_file(
            server_path,
            producer.render_nginx_template(self, self.get_source_path("templates", "server.conf"), site),
        )
        utils.write_file(fragment_path, producer.render_nginx_template(self, source, site))
        if auth:
            utils.write_file(
                fragment_path.parent / "00-auth-location.conf",
                producer.render_nginx_template(
                    self, self.get_source_path("templates", "auth_location.conf"), site),
            )

    def _update_files(self) -> None:
        utils.clear_directory(self.get_app_path("conf.d"))
        snippets_path = self.get_app_path("conf.d", "snippets")
        snippets_path.mkdir(parents=True, exist_ok=True)
        utils.write_file(
            self.get_app_path("conf.d", "00-cntr-upgrade.conf"),
            "map $http_upgrade $connection_upgrade { default upgrade; '' close; }\n"
            'geo $cntr_dollar { default "$"; }\n'
            'map $realip_remote_addr $cntr_actual_socket {\n'
            '    "" $remote_addr;\n'
            '    default $realip_remote_addr;\n'
            '}\n',
        )
        utils.write_file(
            self.get_app_path("conf.d", "01-cntr-health.conf"),
            'server { listen unix:/run/nginx-cntr-health.sock; '
            'location = /__cntr/health { default_type text/plain; return 200 "cntr"; } }\n',
        )
        for name in ("header.conf", "header_all.conf", "params.conf", "auth.conf"):
            self.render_template(
                self.get_source_path("templates", name),
                snippets_path / name,
            )

        domains = {}
        explicit_default = False
        for producer, local_id, site in self.manager.iter_integrations("nginx"):
            if not isinstance(site, NginxSite):
                raise ContainerError(
                    f"Invalid nginx site {producer.name}/{local_id}: expected NginxSite")
            domain = str(site.server_name)
            if not domain:
                continue
            key = domain.lower()
            if key in domains:
                raise ContainerError(
                    f"Duplicate nginx server_name {domain!r}: "
                    f"{domains[key]} and {(producer.name, local_id)}")
            domains[key] = (producer.name, local_id)
            explicit_default |= domain == "_"
            self._write_site(producer, local_id, site)
        if not explicit_default:
            self._write_site(
                self, "default",
                NginxSite(
                    server_name="_",
                    template=self.get_source_path("templates", "index.conf"),
                    https=self.get_config("NGINX_HTTPS_ENABLE", type=bool),
                    waf=False, auth=False,
                ),
            )

    def on_started(self, context: "EventContext") -> None:
        # 更新证书（如果启用HTTPS）
        if self.get_config("NGINX_HTTPS_ENABLE"):
            self.logger.info("Renew nginx certificates if necessary.")
            self.runtime.create_docker_process(
                "exec", "-it", self.get_service_name("nginx"),
                "sh", "-c", f"acme.sh --renew --issue "
                            f"{self.acme_ssl_domains_args} "
                            f"--dns {self.get_config('ACME_DNS_API')} "
                            f"1>/dev/null"
            ).call()
            self.runtime.create_docker_process(
                "exec", "-it", self.get_service_name("nginx"),
                "sh", "-c", f"acme.sh --install-cert "
                            f"{self.acme_ssl_domains_args} "
                            f"{self.acme_ssl_certificate_args} "
                            f"1>/dev/null"
            ).call()

        self.runtime.create_docker_process(
            "exec", self.get_service_name("nginx"), "nginx", "-t"
        ).check_call()
        self.runtime.create_docker_process(
            "exec", self.get_service_name("nginx"), "nginx", "-s", "reload"
        ).check_call()

    def on_stopped(self, context: "EventContext") -> None:
        if context.is_full_containers:
            self.on_removed(context)
            return
        for container in context.target_containers:
            path = self.get_app_path("temporary", container.name)
            if path.exists():
                utils.remove_file(path)

    def on_removed(self, context: "EventContext") -> None:
        pass
