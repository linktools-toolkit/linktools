#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared test harness for linktools-cntr snapshot tests.

Builds a deterministic, non-interactive :class:`ContainerManager` over a
temporary data directory so builtin/fixture compose output can be rendered and
locked as a regression baseline.

Test-only. It does three things, none of which touch production code:

1. Replaces ``linktools.rich`` prompt/choose/confirm with deterministic fakes so
   rendering never blocks on interaction.
2. Resets the ``global_config`` class cache and points ``LINKTOOLS_*`` at a temp
   root so every derived path (data/temp/cache/config) is isolated per test run.
3. Pre-fills config so rendering is deterministic (the live config system is used
   unchanged; ``cast="path"`` fields now resolve correctly in core).
"""
import getpass
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from linktools.errors import CliError
from linktools.types import MISSING
import linktools.rich as _rich

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Type
    from linktools.cntr.container import BaseContainer

_INTERACTIVE_PATCHED = False


def _placeholder(prompt, type=str, default=MISSING, choices=None, **kwargs):
    """Deterministic stand-in for ``linktools.rich.prompt`` (never blocks).

    Mirrors ``rich.prompt``'s own non-interactive (``_no_input``) behaviour:
    return the default if one is available, otherwise raise -- never fabricate
    a type-appropriate value (0/False/"snapval"). A bare (non-chain) provider
    with no default genuinely has nothing sensible to resolve to; a
    ``ConfigField.chain(...)`` provider relies on exactly this raise to fall
    through to its field-level default, so faking a value here would mask
    that fallback path never being reached in production either.
    """
    # DOCKER_USER must be a real account: DOCKER_UID/DOCKER_GID derive from it
    # via get_uid/get_gid, which fail on synthetic names.
    if prompt == "DOCKER_USER":
        return os.environ.get("SUDO_USER") or getpass.getuser() or "root"
    # Fields whose ``default`` is a fresh utils.random_string()/random_secret()
    # each process (e.g. cached=True password generation) would otherwise make
    # snapshots flake between runs -- pin them to a fixed placeholder instead
    # of falling through to the caller-supplied (non-deterministic) default.
    if prompt in ("LLDAP_ADMIN_PASSWORD",):
        return "snapval"
    if default is not MISSING:
        return default
    if choices:
        return choices[0]
    raise CliError(f"prompt requires interaction but no-input mode is active: {prompt}")


def _placeholder_choose(prompt, choices, **kwargs):
    if isinstance(choices, dict):
        return next(iter(choices))
    return choices[0]


def install_deterministic_interaction() -> None:
    """Globally replace interactive prompt/choose/confirm with deterministic fakes.

    Idempotent. Must run before builtin container modules bind ``prompt``
    (they do ``from linktools.rich import prompt`` at import time).
    """
    global _INTERACTIVE_PATCHED
    if _INTERACTIVE_PATCHED:
        return
    _rich.prompt = _placeholder
    _rich.choose = _placeholder_choose
    _rich.confirm = lambda prompt, default=False, **kw: default
    _INTERACTIVE_PATCHED = True


@lru_cache(maxsize=None)
def builtin_module(name: str) -> "ModuleType":
    """Load a trusted builtin asset with deterministic interaction."""
    install_deterministic_interaction()
    assets = Path(__file__).resolve().parents[2] / "linktools-cntr/src/linktools/assets/containers"
    spec = importlib.util.spec_from_file_location("test_generated_" + name.replace("-", "_"),
                                                 str(assets / name / "container.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def builtin_container_type(name: str) -> "Type[BaseContainer]":
    return builtin_module(name).Container


def _reset_global_config() -> None:
    """Force ``global_config`` to re-read ``LINKTOOLS_*`` on next access.

    ``global_config`` is a class-level cached property shared across all Environ
    instances; if a prior test (or import) accessed it, it is frozen with the old
    paths. Resetting it lets a fresh Environ pick up our temp root.
    """
    from linktools.core._environ import BaseEnviron
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def make_manager(data_path, temp_path, name: str = "aio"):
    """Build a fully-prepared ContainerManager over the given temp dirs.

    Args:
        data_path: directory used as ``LINKTOOLS_DATA_PATH``.
        temp_path: directory used as ``LINKTOOLS_TEMP_PATH``.
        name: compose project name.

    Returns:
        A ContainerManager with every discovered container installed and
        prepared (``prepare_installed_containers`` already run).
    """
    install_deterministic_interaction()
    _reset_global_config()

    data_path = str(data_path)
    temp_path = str(temp_path)
    storage = os.path.dirname(data_path) or data_path
    os.environ["LINKTOOLS_PATH"] = storage
    os.environ["LINKTOOLS_DATA_PATH"] = data_path
    os.environ["LINKTOOLS_TEMP_PATH"] = temp_path

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    environ = Environ()  # fresh instance; no stale instance-level caches
    manager = ContainerManager(environ, name=name)
    manager.installed_state.add(*manager.containers.keys())
    manager.prepare_installed_containers()
    # Read-only plans require configured inputs; preparation does not render
    # Dockerfiles or freeze integration snapshots just to populate defaults.
    for key in ("DOCKER_USER", "DOCKER_TYPE", "DOCKER_APP_PATH", "DOCKER_USER_DATA_PATH",
                "NGINX_ROOT_DOMAIN", "NGINX_HTTP_PORT", "NGINX_HTTPS_ENABLE", "ACME_DNS_API",
                "LLDAP_ADMIN_PASSWORD"):
        manager.env_config.get(key)
    for field in manager.containers["nginx"].extend_configs.values():
        manager.containers["nginx"].get_config(field)
    return manager


def _scrub_pairs(manager):
    """(value, token) pairs to scrub from snapshot text, longest value first."""
    from linktools.capabilities.cntr import __cap_cntr__

    docker_uid = manager.env_config.get("DOCKER_UID", default=None)
    docker_gid = manager.env_config.get("DOCKER_GID", default=None)
    docker_identity = None
    if docker_uid is not None and docker_gid is not None:
        docker_identity = "%s:%s" % (docker_uid, docker_gid)

    raw = [
        ("<DOCKER_UID>:<DOCKER_GID>", docker_identity),
        ("<APP_DATA>", getattr(manager, "app_data_path", None)),
        ("<APP>", getattr(manager, "app_path", None)),
        ("<USER_DATA>", manager.env_config.get("DOCKER_USER_DATA_PATH", default=None)),
        ("<DOWNLOAD>", manager.env_config.get("DOCKER_DOWNLOAD_PATH", default=None)),
        ("<DATA>", str(manager.environ.data_path)),
        ("<TEMP>", str(manager.environ.temp_path)),
        ("<ASSETS>", str(__cap_cntr__.get_asset_path("containers"))),
    ]
    pairs = [(str(value), token) for token, value in raw if value]
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def normalize_compose(data, manager) -> str:
    """Render-independent normalized JSON of a compose dict (test-only).

    Eliminates key-order / whitespace differences and scrubs environment-specific
    absolute paths and host identity values so committed snapshots stay portable
    across machines and install layouts.
    """
    text = json.dumps(
        yaml.safe_load(yaml.safe_dump(data, sort_keys=False, allow_unicode=True)),
        sort_keys=True, ensure_ascii=False, indent=2,
    )
    for value, token in _scrub_pairs(manager):
        text = text.replace(value, token)
    return text


def stub_generated_runtime(manager, monkeypatch):
    """Make command-routing tests independent of a native Docker daemon."""
    from types import SimpleNamespace
    from linktools.cntr.runtime.inspect import ProjectRuntimeState

    monkeypatch.setattr(manager.docker_inspector, "get_project_state", lambda containers:
                        ProjectRuntimeState(manager.project_name, (), "docker"))
    monkeypatch.setattr(manager.compose_runner, "wait_service_ready",
                        lambda context, service, **kwargs: True)
    monkeypatch.setattr(manager.compose_runner, "validate_service", lambda *args, **kwargs:
                        SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0))
    monkeypatch.setattr(manager.image_preparer, "image_id", lambda image: "sha256:local-" + image)
    monkeypatch.setattr(manager.image_preparer, "verify_builds", lambda model, services: None)
