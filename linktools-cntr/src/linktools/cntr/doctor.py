#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only environment & security checks.

``ct-cntr doctor`` inspects the runtime, generated compose, repos and config and
reports findings as [WARN]/[INFO]/[OK]. It never modifies anything -- every new
or safer behavior stays opt-in elsewhere.
"""
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dulwich.errors import NotGitRepository

if TYPE_CHECKING:
    from collections.abc import Iterable
    from .container import BaseContainer
    from .manager import ContainerManager


WARN = "WARN"
INFO = "INFO"
OK = "OK"


@dataclass
class Finding:
    severity: str
    message: str


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
                f"{where} uses image tag `latest`. Kept for compatibility; "
                f"consider pinning an explicit tag."))

        for volume in config.get("volumes") or []:
            text = str(volume)
            if "docker.sock" in text or "/var/run/docker" in text:
                findings.append(Finding(
                    WARN,
                    f"{where} mounts the docker socket ({text}). This grants high "
                    f"privilege over the host container runtime."))

        for entry in _env_entries(config.get("environment")):
            key, _, value = entry.partition("=")
            if key == "NODE_TLS_REJECT_UNAUTHORIZED" and str(value).strip() == "0":
                findings.append(Finding(
                    WARN,
                    f"{where} sets NODE_TLS_REJECT_UNAUTHORIZED=0, disabling TLS "
                    f"verification. Kept for compatibility; consider removing it."))
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
            "podman": shutil.which("podman"),
        }
        wanted = runtimes.get(container_type)
        if not wanted:
            findings.append(Finding(WARN, f"container type is `{container_type}` but its "
                                          f"runtime binary was not found on PATH."))
        else:
            findings.append(Finding(OK, f"runtime binary for `{container_type}` found."))

        host = self.manager.container_host
        if host and ("docker.sock" in host or host.endswith(".sock")):
            if not os.path.exists(host):
                findings.append(Finding(WARN, f"DOCKER_HOST socket `{host}` does not exist."))
        return findings

    def check_compose(self, containers: "Iterable[BaseContainer]") -> "list[Finding]":
        findings: "list[Finding]" = []
        seen_container_names: "dict[str, str]" = {}
        for container in containers:
            try:
                compose = container.docker_compose
            except Exception as exc:  # noqa: BLE001 - doctor must never crash on one container
                findings.append(Finding(WARN, f"failed to render compose for "
                                              f"`{container.name}`: {exc}"))
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
                            f"({owner} vs {container.name}/{service})."))
                    else:
                        seen_container_names[cname] = f"{container.name}/{service}"
        return findings

    def check_repos(self) -> "list[Finding]":
        findings: "list[Finding]" = []
        from linktools.git import GitRepository
        for url, meta in self.manager.get_all_repos().items():
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path):
                continue
            if os.path.islink(repo_path):
                findings.append(Finding(INFO, f"repo `{url}` is a local symlink ({repo_path})."))
                continue
            try:
                repo = GitRepository(self.manager.environ, repo_path)
            except NotGitRepository:
                continue
            try:
                if repo.is_dirty():
                    findings.append(Finding(INFO, f"repo `{url}` has uncommitted changes."))
            except Exception:  # noqa: BLE001
                pass
        return findings

    def run(self) -> "list[Finding]":
        findings = self.check_runtime()
        findings.extend(self.check_repos())
        try:
            containers = self.manager.prepare_installed_containers()
            findings.extend(self.check_compose(containers))
        except Exception as exc:  # noqa: BLE001 - doctor must stay read-only & non-fatal
            findings.append(Finding(INFO, f"skipped compose checks: {exc}"))
        return findings
