#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx reverse-proxy container definition."""
import json
import os
import re
import shlex
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, PrefixLoader, StrictUndefined, TemplateError

from linktools import utils
from linktools.cntr import BaseContainer, ContainerError
from linktools.cntr.container import ContainerTemplateError
from linktools.core import ConfigField, PromptProvider, LazyProvider, AliasProvider, ConfirmProvider
from linktools.decorator import cached_property
from linktools.errors import ConfigNotFoundError
from linktools.rich import choose
from linktools.types import MISSING

if TYPE_CHECKING:
    from types import SimpleNamespace
    from linktools.cntr.ext import ResolvedSite
    from collections.abc import Iterable, Sequence
    from typing import AbstractSet, Any, Mapping
    from linktools.cntr import OperationContext
    from linktools.types import PathType


class Container(BaseContainer):

    def quote(self, value: object) -> str:
        data = str(value)
        if any(ch in data for ch in ("\r", "\n", "\x00")):
            raise ContainerError("Nginx header value contains a control character")
        data = data.replace("\\", "\\\\").replace('"', '\\"')
        return '"' + data.replace("$", "$" + "{literal_dollar}") + '"'

    @staticmethod
    def _validated_auth_headers(headers: "Mapping[str, str]") -> "dict[str, str]":
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
            if normalized in seen or normalized in forbidden or normalized.startswith("x-proxy-original-"):
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
            ACME_SERVER="letsencrypt",
            ACME_ACCOUNT_EMAIL="",
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

    @cached_property
    def acme_ssl_domains(self) -> "list[str]":
        result = []
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            result.extend([domain, f"*.{domain}"])
        if self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            for site in self.sites.values():
                if not site.enabled or not site.https:
                    continue
                for value in site.cert_domains:
                    if value and value not in result:
                        result.append(value)
        return result

    @cached_property
    def cert_image_revision(self) -> str:
        """Identify build code and TLS inputs without including DNS credentials."""
        parts = [self.get_config("NGINX_TAG"),
                 str(self.get_config("NGINX_HTTPS_ENABLE", type=bool)),
                 self.get_source_path("Dockerfile").read_text(encoding="utf-8")]
        if self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            parts.extend((self.get_config("ACME_SERVER"), self.get_config("ACME_DNS_API"),
                          self.get_config("ACME_ACCOUNT_EMAIL")))
            parts.extend(self.acme_ssl_domains)
            parts.extend(self.get_source_path(name).read_text(encoding="utf-8")
                         for name in ("nginx-certificates", "nginx-acme", "dnsapi.json"))
        return hashlib.sha256("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    def get_build_revision(self, service: str) -> "str | None":
        return self.cert_image_revision if service == "nginx" else None

    @cached_property
    def acme_ssl_domains_args(self) -> str:
        return " ".join("--domain " + shlex.quote(domain) for domain in self.acme_ssl_domains if domain)

    @cached_property
    def acme_ssl_certificate_args(self) -> str:
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            return " ".join([
                "--cert-file", shlex.quote(f"/etc/certs/{domain}_cert.pem"),
                "--key-file", shlex.quote(f"/etc/certs/{domain}_key.pem"),
                "--fullchain-file", shlex.quote(f"/etc/certs/{domain}_fullchain.pem"),
            ])
        return ""

    def shell_quote(self, value: object) -> str:
        return shlex.quote(str(value))

    def _get_default_index_url(self):
        return utils.make_url(
            self.get_config("NGINX_DEFAULT_SCHEME"),
            "www.google.com" \
                if self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost") \
                else self.get_config("NGINX_ROOT_DOMAIN"),
            self.get_config("NGINX_DEFAULT_PORT")
        )

    def on_check(self, context: "OperationContext") -> None:
        import shutil
        import tempfile
        if self.get_config("NGINX_WILDCARD_DOMAIN") and self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost"):
            raise ContainerError("Wildcard domain is enabled but root domain is not set.")
        if self.get_config("NGINX_WAF_ENABLE") and not self.containers["safeline"].enable:
            raise ContainerError("NGINX_WAF_ENABLE is true but safeline container is not enabled.")
        if self.get_config("NGINX_AUTH_ENABLE") and not self.containers["authelia"].enable:
            raise ContainerError("NGINX_AUTH_ENABLE is true but authelia container is not enabled.")
        runner = self.manager.compose_runner
        command = ("nginx", "-p", "/etc/nginx/", "-c", "/etc/nginx/cntr/nginx.conf", "-t")
        if self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            with tempfile.TemporaryDirectory(prefix="cntr-nginx-check-") as directory:
                source = self.get_app_path("certs", self.cert_image_revision, "live").resolve()
                previous = Path(directory) / self.cert_image_revision / "previous"
                previous.mkdir(parents=True, mode=0o700)
                domain = self.get_config("NGINX_ROOT_DOMAIN")
                for suffix in ("cert", "key", "fullchain"):
                    path = source / (domain + "_" + suffix + ".pem")
                    if path.is_file():
                        shutil.copy2(str(path), str(previous / path.name))
                result = runner.validate_service(context, "nginx", (
                    "/bin/sh", "-c", "/usr/local/bin/nginx-certificates check && "
                    "exec nginx -p /etc/nginx/ -c /etc/nginx/cntr/nginx.conf -t"),
                    mount_overrides={"/etc/certs": directory}, check=False)
        else:
            result = runner.validate_service(context, "nginx", command, check=False)
        if not result.succeeded or "conflicting server name" in (result.stdout + result.stderr).lower():
            match = re.search(r" in ([/A-Za-z0-9_.-]+):(\d+)", result.stderr)
            location = " at {}:{}".format(*match.groups()) if match else ""
            raise ContainerError("Native validation failed for nginx{} (exit {})".format(
                location, result.returncode))
    @cached_property
    def sites(self) -> "Mapping[tuple[str, str], ResolvedSite]":
        return self.manager.nginx_sites

    def complex_value(self, value: str) -> str:
        """Quote one native nginx complex value without interpreting its variables."""
        if not isinstance(value, str) or any(ch in value for ch in ("\r", "\n", "\x00")):
            raise ContainerError("Invalid nginx complex value")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def header_items(self, site: "ResolvedSite | SimpleNamespace", overrides: "Mapping[str, str | None] | None" = None,
                     authentication: bool = False) -> "tuple[tuple[str, str], ...]":
        """Build a complete case-insensitive header set for one proxy location."""
        headers = {
            "Host": "$original_host", "Upgrade": "$http_upgrade",
            "Connection": "$connection_upgrade", "X-Real-IP": "$original_client_ip",
            "X-Original-URL": "$original_scheme://$original_host$original_uri",
            "X-Original-Method": "$original_method", "X-Forwarded-Proto": "$original_scheme",
            "X-Forwarded-Host": "$original_host", "X-Forwarded-URI": "$original_uri",
            "X-Forwarded-Method": "$original_method", "X-Forwarded-For": "$original_client_ip",
            "X-Forwarded-Port": "$original_port", "Forwarded": "",
        }
        for name in ("Scheme", "Host", "URI", "Method", "Client-IP"):
            headers["X-Proxy-Original-" + name] = ""
        for name in ("User", "Groups", "Name", "Email"):
            headers["X-Auth-" + name] = (
                "$identity_" + site.var_name + "_" + name.lower()
                if site.auth and not authentication else "")
        auth_headers = self._validated_auth_headers(site.auth_headers) if site.auth else {}
        if not authentication:
            for index, key in enumerate(auth_headers):
                headers[key] = "$credential_" + site.var_name + "_" + str(index)
        seen = set()
        protected = {key.lower() for key in auth_headers}
        for key, value in (overrides or {}).items():
            if not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key):
                raise ContainerError("Invalid nginx header override name")
            normalized = key.lower()
            if (normalized in seen or normalized.startswith("x-proxy-original-") or
                    normalized in {"x-auth-user", "x-auth-groups", "x-auth-name", "x-auth-email"} or
                    (not authentication and normalized in protected)):
                raise ContainerError("Duplicate or reserved nginx header override: " + key)
            seen.add(normalized)
            value = "" if value is None else value
            self.complex_value(value)
            for original in tuple(headers):
                if original.lower() == normalized:
                    del headers[original]
            headers[key] = value
        if authentication:
            headers.update({"Content-Length": "", "Connection": "", "Upgrade": ""})
        return tuple((key, self.complex_value(value)) for key, value in headers.items())

    def security_maps(self, site: "ResolvedSite | SimpleNamespace") -> str:
        """Generate only maps whose inputs and evaluation phase are explicit."""
        lines = []
        for capability in ("waf", "auth"):
            if not getattr(site, capability):
                continue
            lines.extend(["map $uri $" + capability + "_skip_" + site.var_name + " {", "    default 0;"])
            for regex in getattr(site, capability + "_bypass"):
                lines.append("    " + self.complex_value("~*" + regex) + " 1;")
            lines.append("}")
        if not site.auth:
            return "\n".join(lines)
        suffix = site.var_name
        # Exact keys preserve business regex captures when evaluated at proxy time.
        lines.append('map "$auth_status_' + suffix + ':$auth_skip_' + suffix + ':$auth_proof_' + suffix + '" $auth_verified_' + suffix + ' {')
        lines.append("    volatile;")
        lines.append("    default 0;")
        lines.extend('    "' + str(status) + ':0:1" 1;' for status in range(200, 300))
        lines.append("}")
        for name in ("user", "groups", "name", "email"):
            lines.extend([
                "map $auth_verified_" + suffix + " $identity_" + suffix + "_" + name + " {",
                '    volatile;', '    default "";', "    1 $auth_" + name + "_" + suffix + ";", "}",
            ])
        for index, (key, value) in enumerate(self._validated_auth_headers(site.auth_headers).items()):
            # nginx ignores non-conventional incoming header names by default;
            # its $http_* variable syntax cannot address their punctuation.
            incoming = ("${http_" + key.lower().replace("-", "_") + "}"
                        if re.fullmatch(r"[A-Za-z0-9_-]+", key) else '""')
            lines.extend([
                "map $auth_verified_" + suffix + " $credential_" + suffix + "_" + str(index) + " {",
                "    volatile;", "    default " + incoming + ";", "    1 " + self.quote(value) + ";", "}",
            ])
        return "\n".join(lines)


    def get_runtime_requirements(self, required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
        manager = self.manager
        sites = self.sites
        needs_nginx = self.name in required or any(
            producer in required and site.enabled for (producer, _), site in sites.items())
        if not needs_nginx:
            return {}
        result = {self.name: tuple(self.services)}
        for site in sites.values():
            if not site.enabled:
                continue
            for capability, provider in (("auth", "authelia"), ("waf", "safeline")):
                if getattr(site, capability):
                    result[provider] = (("authelia",) if provider == "authelia"
                                        else tuple(manager.containers[provider].services))
        return result

    def _render_site_template(self, container: "BaseContainer", source: "PathType",
                        site: "ResolvedSite | SimpleNamespace", business: "str | None" = None,
                        sites: "Sequence[ResolvedSite] | None" = None,
                        route_auth: bool = False) -> str:
        """Render one location template with unambiguous local/nginx namespaces."""
        nginx = self
        source = Path(source).absolute()
        nginx_root = Path(nginx.get_source_path("templates")).absolute()
        environment = Environment(
            loader=PrefixLoader({
                "local": FileSystemLoader(str(source.parent)),
                "nginx": FileSystemLoader(str(nginx_root)),
            }),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=nginx_root in source.parents,
            lstrip_blocks=nginx_root in source.parents,
        )
        try:
            template_name = "nginx/" + source.relative_to(nginx_root).as_posix()
        except ValueError:
            template_name = "local/" + source.name
        extra = {} if business is None else {"business": business}
        if sites is not None:
            extra["sites"] = sites
        if route_auth:
            extra["route_auth"] = True
        try:
            return environment.get_template(template_name).render(
                site=site,
                container=container,
                nginx=nginx,
                config=container.env_config,
                vars=site.vars,
                **extra,
            )
        except TemplateError as exc:
            raise ContainerTemplateError(
                f"Invalid nginx template {source} for {container.name}/{getattr(site, 'local_id', '?')}: {exc}"
            ) from exc

    @cached_property
    def _rendered_site_files(self) -> "tuple[dict[str, str], bool]":
        """Evaluate business templates once for this declaration snapshot."""
        from types import SimpleNamespace
        result = {}
        active = []
        for site in self.sites.values():
            if site.enabled:
                active.append(site.resolve())
        defaults = {}
        for site in active:
            if not site.default:
                continue
            ports = [site.producer.get_config("NGINX_HTTP_PORT")]
            if site.https:
                ports.append(site.producer.get_config("NGINX_HTTPS_PORT"))
            if site.waf:
                ports.append(site.producer.get_config("NGINX_WAF_PORT"))
            for port in ports:
                if port in defaults:
                    raise ContainerError("Multiple default nginx sites on port {}: {}/{} and {}/{}".format(
                        port, defaults[port].producer.name, defaults[port].local_id,
                        site.producer.name, site.local_id))
                defaults[port] = site
        if not any(site.default for site in active):
            active.append(SimpleNamespace(
                producer=self, local_id="default", file_id="default", var_name="default",
                server_name='""', default=True, https=self.get_config("NGINX_HTTPS_ENABLE", type=bool),
                waf=False, auth=False, waf_bypass=(), auth_bypass=(), auth_headers={}, vars={},
                template=self.get_source_path("templates", "index.conf"), proxy=None,
            ))
        from collections import OrderedDict
        groups = OrderedDict()
        for site in active:
            domain = site.server_name
            host_pattern = r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
            if re.fullmatch(host_pattern, domain) or re.fullmatch(r"\*\." + host_pattern, domain):
                domain = domain.lower()
            key = (site.producer.get_config("NGINX_HTTP_PORT"), domain)
            groups.setdefault(key, []).append(site)
        for (port, domain), sites in groups.items():
            leader = sites[0]
            def routing_policy(site):
                return (
                    site.https, site.waf, site.waf_bypass,
                    site.producer.get_config("NGINX_HTTPS_PORT") if site.https else None,
                    site.producer.get_config("NGINX_WAF_PORT") if site.waf else None,
                )
            policy = routing_policy(leader)
            for site in sites[1:]:
                if routing_policy(site) != policy:
                    raise ContainerError(
                        "Incompatible nginx routing policies on {}:{} for {}/{} and {}/{}".format(
                            domain, port, leader.producer.name, leader.local_id,
                            site.producer.name, site.local_id))
            route_auth = len(sites) > 1 and any(
                (member.auth, member.auth_bypass) != (leader.auth, leader.auth_bypass)
                for member in sites[1:])
            leader = next((site for site in sites if site.default), leader)
            routes = []
            for site in sites:
                source = site.template or self.get_source_path("templates", "default.conf")
                business = self._render_site_template(
                    site.producer, source, site, route_auth=route_auth)
                if len(sites) > 1:
                    business = "# site {}/{}\n{}".format(site.producer.name, site.local_id, business)
                routes.append(business)
            result["sites/" + leader.file_id + ".conf"] = self._render_site_template(
                leader.producer, self.get_source_path("templates", "server.conf"),
                leader, business="\n".join(routes), sites=sites, route_auth=route_auth)
        return result, any(site.waf for site in active)

    def on_starting(self, context: "OperationContext") -> None:
        import tarfile
        import tempfile
        from filelock import FileLock
        from types import SimpleNamespace
        from linktools.cntr.artifacts import atomic_write_text_if_changed

        if self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            for name in ("certs", "acme", "acme-secrets"):
                self.get_app_path(name).mkdir(parents=True, exist_ok=True)
            self.runtime.chmod(self.get_app_path("acme-secrets"), 0o700)
            values = []
            for key, field in self.extend_configs.items():
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise ContainerError("Invalid DNS environment variable name: " + key)
                value = str(self.get_config(field))
                if "\n" in value or "\r" in value:
                    raise ContainerError("DNS credentials must be single-line values")
                values.append("export {}={}\n".format(key, shlex.quote(value)))
            atomic_write_text_if_changed(self.get_app_path("acme-secrets", "dns.env"),
                                         "".join(values), mode=0o600)
            lock_path = self.get_app_path("certs", ".acme.lock")
            with FileLock(str(lock_path)):
                self.runtime.chmod(lock_path, 0o600)
                if ("nginx" in context.initial_existing_services and
                        not os.path.lexists(str(self.get_app_path("generated", "current")))):
                    self._preserve_legacy_files()
                account = self.get_app_path("certs", self.cert_image_revision, "live", "acme")
                previous_revision = next((item.labels.get("io.linktools.nginx.certificate-revision")
                                          for item in context.initial_runtime_state.services
                                          if item.service == "nginx"), None)
                if not account.is_dir() and previous_revision is not None:
                    if not re.fullmatch(r"[0-9a-f]{16}", previous_revision):
                        raise ContainerError("Invalid running nginx certificate revision")
                    account = self.get_app_path("certs", previous_revision, "live", "acme")
                if not account.is_dir():
                    account = self.get_app_path("certs", "live", "acme")
                if not account.is_dir():
                    account = self.get_app_path("acme")
                archive = self.get_app_path("acme-build-account.tar")
                with tempfile.NamedTemporaryFile(dir=str(archive.parent), delete=False) as stream:
                    temporary = stream.name
                    try:
                        with tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as output:
                            for entry in account.iterdir():
                                output.add(str(entry), arcname=entry.name)
                    except BaseException:
                        os.unlink(temporary)
                        raise
                os.replace(temporary, str(archive))
        files, waf = self._rendered_site_files
        result = dict(files)
        root_site = SimpleNamespace(vars={"waf": waf, "site_files": tuple(files)})
        result["nginx.conf"] = self._render_site_template(
            self, self.get_source_path("templates", "nginx.conf"), root_site)
        context.write_files(self, result)
    def _preserve_legacy_files(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path
        backup = self.get_app_path("migration-backup")
        if not backup.exists():
            temporary = Path(tempfile.mkdtemp(
                prefix="migration-backup-", dir=str(self.get_app_path())))
            self.logger.info("Preserve legacy nginx certificates and ACME account before mount migration")
            for source, name in (("/etc/certs/.", "certs"), ("/root/.acme.sh/.", "acme")):
                destination = temporary / name
                destination.mkdir()
                self.runtime.create_docker_process("cp", "{}:{}".format(
                    self.get_service_name("nginx"), source), str(destination)).check_call()
            previous = self.get_app_path("conf.d")
            if previous.exists():
                shutil.copytree(str(previous), str(temporary / "conf.d"), symlinks=True)
            os.rename(str(temporary), str(backup))
        if (self.manager.system in ("darwin", "linux") and self.manager.uid != 0
                and self.manager.container_type == "docker"):
            # sudo docker cp preserves private modes but makes copied files root-owned.
            self.runtime.create_process(
                "chown", "-R", "-h", "{}:{}".format(self.manager.uid, self.manager.gid),
                str(backup), privilege=True,
            ).check_call()
        for name in ("certs", "acme"):
            source = backup / name
            if not source.is_dir():
                raise ContainerError("Legacy nginx migration backup is incomplete")
            for path in source.rglob("*"):
                destination = self.get_app_path(name) / path.relative_to(source)
                if os.path.lexists(destination):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    os.symlink(os.readlink(str(path)), str(destination))
                elif path.is_dir():
                    destination.mkdir()
                else:
                    shutil.copy2(str(path), str(destination))
