#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx generated configuration and application."""
import os
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, PrefixLoader, StrictUndefined, TemplateError
from linktools.decorator import cached_property

from ..container import ContainerError, ContainerTemplateError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext
    from linktools.types import PathType
    from typing import Any


class NginxGeneration:
    """Own the builtin nginx consumer without extending container hooks."""

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    def render_template(self, container: "BaseContainer", source: "PathType",
                        site: "Any") -> str:
        """Render one location template with unambiguous local/nginx namespaces."""
        nginx = self.container
        source = Path(source).absolute()
        nginx_root = Path(nginx.get_source_path("templates")).absolute()
        environment = Environment(
            loader=PrefixLoader({
                "local": FileSystemLoader(str(source.parent)),
                "nginx": FileSystemLoader(str(nginx_root)),
            }),
            undefined=StrictUndefined,
            autoescape=False,
        )
        try:
            template_name = "nginx/" + source.relative_to(nginx_root).as_posix()
        except ValueError:
            template_name = "local/" + source.name
        try:
            return environment.get_template(template_name).render(
                site=site,
                container=container,
                nginx=nginx,
                config=container.env_config,
                vars=site.vars,
            )
        except TemplateError as exc:
            raise ContainerTemplateError(
                f"Invalid nginx template {source} for {container.name}/{getattr(site, 'local_id', '?')}: {exc}"
            ) from exc

    @cached_property
    def _acme_ssl_domains(self) -> "list[str]":
        result = []
        domain = self.container.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            result.extend([domain, f"*.{domain}"])
        if self.container.get_config("NGINX_HTTPS_ENABLE", type=bool):
            for site in self.container.sites.values():
                if not site.enabled or not site.https:
                    continue
                for value in site.cert_domains:
                    if value and value not in result:
                        result.append(value)
        return result

    @cached_property
    def acme_ssl_domains_args(self) -> str:
        return " ".join([f"--domain {domain}" for domain in self._acme_ssl_domains if domain])

    @cached_property
    def acme_ssl_certificate_args(self) -> str:
        domain = self.container.get_config("NGINX_ROOT_DOMAIN")
        if domain:
            return " ".join([
                "--cert-file", f"/etc/certs/{domain}_cert.pem",
                "--key-file", f"/etc/certs/{domain}_key.pem",
                "--fullchain-file", f"/etc/certs/{domain}_fullchain.pem",
            ])
        return ""

    @cached_property
    def _rendered_site_files(self) -> "tuple[dict[str, str], bool]":
        """Evaluate business templates once for this declaration snapshot."""
        from types import SimpleNamespace
        result = {}
        active = []
        for site in self.container.sites.values():
            if site.enabled:
                active.append(site.resolve())
        if not any(site.server_name == "_" for site in active):
            active.append(SimpleNamespace(
                producer=self.container, local_id="default", file_id="cntr_default", var_name="cntr_default",
                server_name="_", https=self.container.get_config("NGINX_HTTPS_ENABLE", type=bool),
                waf=False, auth=False, waf_bypass=(), auth_bypass=(), auth_headers={}, vars={},
                template=self.container.get_source_path("templates", "index.conf"), proxy=None,
            ))
        for site in active:
            producer = site.producer
            source = site.template or self.container.get_source_path("templates", "default.conf")
            result["sites/" + site.file_id + "/business.conf"] = self.render_template(producer, source, site)
            result["sites/" + site.file_id + ".conf"] = self.render_template(
                producer, self.container.get_source_path("templates", "server.conf"), site)
            if site.auth:
                result["sites/" + site.file_id + "/auth.conf"] = self.render_template(
                    producer, self.container.get_source_path("templates", "auth_location.conf"), site)
        return result, any(site.waf for site in active)

    def render(self, generation_id: str) -> "dict[str, str]":
        """Render a generation marker around the immutable business snapshot."""
        from types import SimpleNamespace
        files, waf = self._rendered_site_files
        result = dict(files)
        root_site = SimpleNamespace(vars={
            "generation_id": generation_id, "waf": waf,
            "site_files": tuple(name for name in files if name.count("/") == 1),
        })
        result["nginx.conf"] = self.render_template(
            self.container, self.container.get_source_path("templates", "nginx.conf"), root_site)
        return result

    def prepare(self, context: "EventContext") -> None:
        for name in ("generated", "certs", "acme"):
            self.container.get_app_path(name).mkdir(parents=True, exist_ok=True)
        if not self.container.get_config("NGINX_HTTPS_ENABLE", type=bool):
            return
        import shlex
        domain = self.container.get_config("NGINX_ROOT_DOMAIN")
        certificate = self.container.get_app_path("certs", domain + "_fullchain.pem")
        key = self.container.get_app_path("certs", domain + "_key.pem")
        if ("nginx" in getattr(context, "initial_services", getattr(context, "initial_running", ())) and
                not os.path.lexists(self.container.get_app_path("generated", "current"))):
            self._preserve_legacy_files()
        if certificate.exists() and key.exists():
            checks = ["openssl x509 -checkend 2592000 -noout -in " +
                      shlex.quote("/etc/certs/" + domain + "_fullchain.pem")]
            for name in self._acme_ssl_domains:
                if name.startswith("*."):
                    checks.append(r"openssl x509 -noout -ext subjectAltName -in {} | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fx -- {}".format(
                        shlex.quote("/etc/certs/" + domain + "_fullchain.pem"), shlex.quote("DNS:" + name)))
                else:
                    checks.append("openssl x509 -noout -checkhost {} -in {}".format(
                        shlex.quote(name), shlex.quote("/etc/certs/" + domain + "_fullchain.pem")))
            try:
                self.container.manager.compose_runner.validate_service(context, "nginx", ("sh", "-c", " && ".join(checks)))
                return
            except ContainerError:
                self.container.logger.info("Renew certificate for expiry or changed domain coverage")
        # Existing certificate/account volumes are reused; issuance is never a
        # build step and credentials are never baked into an image layer.
        domains = " ".join("--domain " + shlex.quote(item) for item in self._acme_ssl_domains)
        command = "acme.sh --config-home /root/.acme.sh --issue {} --dns {} && acme.sh --config-home /root/.acme.sh --install-cert {} {}".format(
            domains, shlex.quote(self.container.get_config("ACME_DNS_API")), domains,
            self.acme_ssl_certificate_args)
        self.container.manager.compose_runner.validate_service(
            context, "nginx", ("sh", "-c", command), network=True)

    def _preserve_legacy_files(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path
        backup = self.container.get_app_path("migration-backup")
        if not backup.exists():
            temporary = Path(tempfile.mkdtemp(
                prefix="migration-backup-", dir=str(self.container.get_app_path())))
            self.container.logger.info("Preserve legacy nginx certificates and ACME account before mount migration")
            for source, name in (("/etc/certs/.", "certs"), ("/root/.acme.sh/.", "acme")):
                destination = temporary / name
                destination.mkdir()
                self.container.runtime.create_docker_process("cp", "{}:{}".format(
                    self.container.get_service_name("nginx"), source), str(destination)).check_call()
            previous = self.container.get_app_path("conf.d")
            if previous.exists():
                shutil.copytree(str(previous), str(temporary / "conf.d"), symlinks=True)
            os.rename(str(temporary), str(backup))
        for name in ("certs", "acme"):
            source = backup / name
            if not source.is_dir():
                raise ContainerError("Legacy nginx migration backup is incomplete")
            for path in source.rglob("*"):
                destination = self.container.get_app_path(name) / path.relative_to(source)
                if os.path.lexists(destination):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    os.symlink(os.readlink(str(path)), str(destination))
                elif path.is_dir():
                    destination.mkdir()
                else:
                    shutil.copy2(str(path), str(destination))

    def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        self.container.manager.compose_runner.validate_service(context, "nginx", (
            "nginx", "-p", "/etc/nginx/", "-c",
            "/etc/nginx/generated/{}/nginx.conf".format(candidate.generation_id), "-t"))

    def confirm(self, context: "EventContext", generation_id: str,
                timeout: int = 30) -> None:
        import time
        deadline = time.monotonic() + timeout
        while True:
            result = self.container.manager.compose_runner.exec_service(context, "nginx", (
                "curl", "--fail", "--silent", "--max-time", "2", "--unix-socket",
                "/run/nginx-cntr-health.sock", "http://localhost/__cntr/health"), check=False)
            if result.succeeded and result.stdout.strip() == generation_id:
                return
            if time.monotonic() >= deadline:
                raise ContainerError("Nginx did not acknowledge the generated configuration")
            time.sleep(0.25)

    def bootstrap(self, context: "EventContext") -> None:
        from linktools.cntr.artifacts import GeneratedCandidate
        def render(generation_id: str) -> "dict[str, str]":
            return {"nginx.conf": 'events {}\nhttp {\n'
                    'server { listen unix:/run/nginx-cntr-health.sock; '
                    'location = /__cntr/health { default_type text/plain; return 200 "' + generation_id + '"; }}\n'
                    'server { listen ' + str(self.container.get_config("NGINX_HTTP_PORT")) + ' default_server; return 503; }\n}\n'}
        candidate = GeneratedCandidate(self.container, render=render)
        self.validate(candidate, context)
        candidate.publish()
        self.container.manager.compose_runner.apply_service(context, "nginx", recreate=True)
        self.confirm(context, candidate.generation_id)
        context.nginx_bootstrap_id = candidate.generation_id

    def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
              services: "Iterable[str]") -> None:
        if "nginx" not in services:
            return
        runner = self.container.manager.compose_runner
        # This also handles the first stable-parent mount and target-image
        # changes, using Compose's ordinary reconciliation.
        runner.apply_service(context, "nginx")
        result = runner.exec_service(context, "nginx", (
            "curl", "--fail", "--silent", "--max-time", "2", "--unix-socket",
            "/run/nginx-cntr-health.sock", "http://localhost/__cntr/health"), check=False)
        if result.succeeded and result.stdout.strip() == candidate.generation_id:
            return
        runner.exec_service(context, "nginx", (
            "nginx", "-p", "/etc/nginx/", "-c", "/etc/nginx/generated/current/nginx.conf", "-s", "reload"))
        self.confirm(context, candidate.generation_id)
