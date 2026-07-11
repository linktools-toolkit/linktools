#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only environment & security checks.

``ct-cntr doctor`` inspects the runtime, generated compose, repos and
config, and reports findings as [ERROR]/[WARN]/[INFO]/[OK]. It never
modifies anything -- every new or safer behavior stays opt-in elsewhere.
"""
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dulwich.errors import NotGitRepository

from ..capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from collections.abc import Iterable
    from .container import BaseContainer
    from .manager import ContainerManager


WARN = "WARN"
INFO = "INFO"
OK = "OK"

# Stable finding codes -- part of the --json contract; don't rename once
# released.
RUNTIME_BINARY_MISSING = "runtime.binary_missing"
RUNTIME_ENDPOINT_MISSING = "runtime.endpoint_missing"
RUNTIME_ACCESS_DENIED = "runtime.access_denied"
COMPOSE_VALIDATION_FAILED = "compose.validation_failed"
REPO_CONFIG_INVALID = "repo.config_invalid"
REPO_INCOMPATIBLE = "repo.incompatible"
REPO_DIRTY = "repo.dirty"
ARTIFACT_STALE = "artifact.stale"
SECURITY_DOCKER_SOCKET_MOUNT = "security.docker_socket_mount"
SECURITY_LATEST_IMAGE = "security.latest_image"
SECURITY_TLS_DISABLED = "security.tls_disabled"


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str
    code: "str | None" = None
    component: "str | None" = None
    details: "dict[str, object]" = field(default_factory=dict)


def _image_uses_latest(image: str) -> bool:
    """True if ``image`` is untagged or pinned to ``latest``."""
    if not image:
        return False
    # strip digest / registry port noise: tag is after the last ':' that is not part of a port
    tail = image.rsplit("/", 1)[-1]
    if ":" not in tail:
        return True  # no tag -> implicit latest
    return tail.rsplit(":", 1)[-1] == "latest"


def _env_entries(environment: "Any") -> "list[str]":
    """Normalize a compose service ``environment`` block to ``KEY=value`` strings."""
    if not environment:
        return []
    if isinstance(environment, dict):
        return [f"{k}={v}" for k, v in environment.items()]
    if isinstance(environment, list):
        return [str(e) for e in environment]
    return []


def scan_compose(container_name: str, compose: "dict[str, Any] | None") -> "list[Finding]":
    """Inspect one container's rendered compose for risk signals (read-only)."""
    findings: "list[Finding]" = []
    services = (compose or {}).get("services") or {}
    for service, config in services.items():
        if not isinstance(config, dict):
            continue
        where = f"{container_name}/{service}"

        image = config.get("image")
        if isinstance(image, str) and _image_uses_latest(image):
            findings.append(Finding(
                WARN,
                f"{where} uses image tag `latest`. Consider pinning an explicit tag.",
                code=SECURITY_LATEST_IMAGE, component=where))

        for volume in config.get("volumes") or []:
            text = str(volume)
            if "docker.sock" in text or "/var/run/docker" in text:
                findings.append(Finding(
                    WARN,
                    f"{where} mounts the docker socket ({text}). This grants high "
                    f"privilege over the host container runtime.",
                    code=SECURITY_DOCKER_SOCKET_MOUNT, component=where))

        for entry in _env_entries(config.get("environment")):
            key, _, value = entry.partition("=")
            if key == "NODE_TLS_REJECT_UNAUTHORIZED" and str(value).strip() == "0":
                findings.append(Finding(
                    WARN,
                    f"{where} sets NODE_TLS_REJECT_UNAUTHORIZED=0, disabling TLS "
                    f"verification. Consider removing it.",
                    code=SECURITY_TLS_DISABLED, component=where))
    return findings


class Doctor:
    """Runs read-only checks against a ContainerManager."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def check_runtime(self) -> "list[Finding]":
        findings: "list[Finding]" = []
        container_type = self.manager.container_type
        runtimes = {
            "docker": shutil.which("docker"),
            "docker-rootless": shutil.which("docker"),
        }
        wanted = runtimes.get(container_type)
        if not wanted:
            findings.append(Finding(
                WARN, f"container type is `{container_type}` but its runtime binary was not found on PATH.",
                code=RUNTIME_BINARY_MISSING, component=container_type))
        else:
            findings.append(Finding(OK, f"runtime binary for `{container_type}` found.", component=container_type))

        host = self.manager.container_host
        if host and ("docker.sock" in host or host.endswith(".sock")):
            if not os.path.exists(host):
                findings.append(Finding(
                    WARN, f"DOCKER_HOST socket `{host}` does not exist.",
                    code=RUNTIME_ENDPOINT_MISSING, component="docker"))

        if wanted:
            findings.extend(self._check_runtime_versions())
        return findings

    def _check_runtime_versions(self) -> "list[Finding]":
        # sudo (when required) blocks for a password interactively; a probe
        # failure (sudo policy denied, daemon unreachable, unparsable
        # output) is reported as WARN with runtime.access_denied, never
        # mistaken for "not installed".
        findings: "list[Finding]" = []
        engine = self.manager.docker_inspector.get_engine_version()
        if engine.server:
            findings.append(Finding(OK, f"docker engine version: {engine.server}.", component="docker"))
        else:
            findings.append(Finding(
                WARN, "docker engine version could not be determined. "
                      "Check sudo policy, or use docker-rootless.",
                code=RUNTIME_ACCESS_DENIED, component="docker"))

        compose_version = self.manager.docker_inspector.get_compose_version()
        if compose_version:
            findings.append(Finding(OK, f"docker compose version: {compose_version}.", component="docker-compose"))
        else:
            findings.append(Finding(
                WARN, "docker compose version could not be determined. "
                      "Check sudo policy, or use docker-rootless.",
                code=RUNTIME_ACCESS_DENIED, component="docker-compose"))
        return findings

    def check_compose(self, containers: "Iterable[BaseContainer]") -> "list[Finding]":
        findings: "list[Finding]" = []
        seen_container_names: "dict[str, str]" = {}
        for container in containers:
            try:
                compose = container.docker_compose
            except Exception as exc:  # noqa: BLE001 - doctor must never crash on one container
                findings.append(Finding(
                    WARN, f"failed to render compose for `{container.name}`: {exc}",
                    code=COMPOSE_VALIDATION_FAILED, component=container.name))
                continue
            findings.extend(scan_compose(container.name, compose))
            for service, config in (compose or {}).get("services", {}).items() if compose else []:
                if not isinstance(config, dict):
                    continue
                cname = config.get("container_name")
                if cname:
                    owner = seen_container_names.get(cname)
                    if owner is not None and owner != f"{container.name}/{service}":
                        findings.append(Finding(
                            WARN,
                            f"duplicate container_name `{cname}` "
                            f"({owner} vs {container.name}/{service}).",
                            component=container.name))
                    else:
                        seen_container_names[cname] = f"{container.name}/{service}"
        return findings

    def check_compose_validation(self, containers: "Iterable[BaseContainer]") -> "list[Finding]":
        """``docker compose config`` validation (only when a runtime binary
        is present -- this genuinely talks to Docker, unlike check_compose's
        static scan)."""
        findings: "list[Finding]" = []
        if not shutil.which("docker"):
            return findings
        containers = tuple(containers)
        if not containers:
            return findings
        try:
            result = self.manager.docker_inspector.validate_compose(containers)
        except Exception as exc:  # noqa: BLE001 - doctor must stay read-only & non-fatal
            findings.append(Finding(WARN, f"compose validation could not run: {exc}", code=COMPOSE_VALIDATION_FAILED))
            return findings
        if not result.succeeded:
            findings.append(Finding(
                WARN, f"docker compose config failed: {result.stderr.strip()[:500]}",
                code=COMPOSE_VALIDATION_FAILED))
        return findings

    def check_repos(self) -> "list[Finding]":
        findings: "list[Finding]" = []
        from linktools.git import GitRepository
        from linktools.core import ensure_requirement
        from linktools.errors import ConfigError, ConfigValidationError
        for url, meta in self.manager.repo_store.get_all().items():
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path):
                continue

            # Same load+gate ContainerLoader/RepoStore.add use before
            # accepting a repo: an invalid .linktools.json is reported, not
            # silently skipped -- Doctor must never report a repo as clean
            # just because it has nothing to check.
            file_config = None
            try:
                file_config = self.manager.environ.load_file_config(local_root=repo_path)
            except ConfigError as exc:
                findings.append(Finding(
                    WARN, f"repo `{url}` has an invalid .linktools.json: {exc}",
                    code=REPO_CONFIG_INVALID, component=url))

            if file_config is not None:
                try:
                    ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
                except ConfigValidationError as exc:
                    findings.append(Finding(
                        WARN, f"repo `{url}` {exc}",
                        code=REPO_INCOMPATIBLE, component=url))

            if os.path.islink(repo_path):
                findings.append(Finding(INFO, f"repo `{url}` is a local symlink ({repo_path}).", component=url))
                continue
            try:
                repo = GitRepository(self.manager.environ, repo_path)
            except NotGitRepository:
                continue
            try:
                if repo.is_dirty():
                    findings.append(Finding(
                        INFO, f"repo `{url}` has uncommitted changes.", code=REPO_DIRTY, component=url))
            except Exception:  # noqa: BLE001
                pass
        return findings

    def check_artifacts(self, containers: "Iterable[BaseContainer]") -> "list[Finding]":
        """Report an indexed artifact whose container no longer produces it.
        Report-only: never deletes the stale index entry."""
        from .artifacts.index import collect_candidates
        findings: "list[Finding]" = []
        indexed = self.manager.artifact_index.load()
        if not indexed:
            return findings
        candidates = collect_candidates(self.manager, tuple(containers))
        current_paths = {
            os.path.relpath(dest, str(self.manager.data_path)).replace(os.sep, "/")
            for dest in candidates
        }
        for path in sorted(set(indexed) - current_paths):
            findings.append(Finding(
                INFO, f"generated artifact `{path}` is no longer produced by any installed container.",
                code=ARTIFACT_STALE, component=path))
        return findings

    def run(self, runtime: bool = False) -> "list[Finding]":
        findings = self.check_runtime()
        findings.extend(self.check_repos())
        try:
            containers = self.manager.prepare_installed_containers()
            findings.extend(self.check_compose(containers))
            findings.extend(self.check_artifacts(containers))
            if runtime:
                findings.extend(self.check_compose_validation(containers))
        except Exception as exc:  # noqa: BLE001 - doctor must stay read-only & non-fatal
            findings.append(Finding(INFO, f"skipped compose checks: {exc}"))
        return findings
