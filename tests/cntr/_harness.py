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
import json
import os

import yaml

from linktools.errors import CliError
from linktools.types import MISSING
import linktools.rich as _rich

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
    manager.add_installed_containers(*manager.containers.keys())
    manager.prepare_installed_containers()
    return manager


def _scrub_pairs(manager):
    """(path, token) pairs to scrub from snapshot text, longest path first."""
    from linktools.capabilities.cntr import __cap_cntr__
    raw = [
        ("<APP_DATA>", getattr(manager, "app_data_path", None)),
        ("<APP>", getattr(manager, "app_path", None)),
        ("<USER_DATA>", manager.env_config.get("DOCKER_USER_DATA_PATH", default=None)),
        ("<DOWNLOAD>", manager.env_config.get("DOCKER_DOWNLOAD_PATH", default=None)),
        ("<DATA>", str(manager.environ.data_path)),
        ("<TEMP>", str(manager.environ.temp_path)),
        ("<ASSETS>", str(__cap_cntr__.get_asset_path("containers"))),
    ]
    pairs = [(str(path), token) for token, path in raw if path]
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def normalize_compose(data, manager) -> str:
    """Render-independent normalized JSON of a compose dict (test-only).

    Eliminates key-order / whitespace differences and scrubs environment-specific
    absolute paths so committed snapshots stay portable across machines and
    install layouts.
    """
    text = json.dumps(
        yaml.safe_load(yaml.safe_dump(data, sort_keys=False, allow_unicode=True)),
        sort_keys=True, ensure_ascii=False, indent=2,
    )
    for path, token in _scrub_pairs(manager):
        text = text.replace(path, token)
    return text
