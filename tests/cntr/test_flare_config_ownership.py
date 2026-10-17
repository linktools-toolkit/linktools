#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing template ownership hooks follow generated-file preparation."""
import os
from pathlib import Path

import pytest

from _harness import configure_builtin_inputs
from linktools.cntr import OperationContext
from linktools.cntr.lifecycle.hooks import HookPhase


@pytest.mark.parametrize("collateral", [False, True])
def test_flare_template_ownership_runs_after_prepare(fresh_manager, monkeypatch, collateral):
    manager = fresh_manager
    configure_builtin_inputs(manager)
    manager.config.set("DOCKER_UID", manager.uid + 10000)
    manager.config.set("DOCKER_GID", manager.gid + 10000)
    flare = manager.containers["flare"]
    spec = flare.services["flare"]
    path = flare.config_path
    assert not path.exists()
    assert "group_add" not in spec
    assert spec["user"] == "{}:{}".format(manager.uid + 10000, manager.gid + 10000)
    assert all(mount.endswith(":ro") for mount in spec["volumes"] if mount.startswith(str(path)))
    calls = []

    def chown(target, user, recursive=False, **kwargs):
        target = Path(target)
        assert target.is_file()
        assert target.stat().st_mode & 0o777 == 0o640
        assert target.read_text(encoding="utf-8")
        calls.append((target.name, user, recursive, kwargs))

    monkeypatch.setattr(manager.runtime, "chown", chown)
    monkeypatch.setattr(manager.runtime, "chmod", lambda target, mode: os.chmod(target, mode))
    if collateral:
        for key, source in ((("ordinary",), "custom"),
                            (("chown", str(path / "other"), "1", False, "2"), "builtin"),
                            (("chown", str(path / "apps.yml"), "1", True, "2"), "builtin"),
                            (("chown", str(path / "apps.yml"), "1", False, "2"), "custom")):
            flare.add_start_hook(key, lambda: pytest.fail("unrelated startup hook"), source=source)
        flare.on_starting = lambda ctx: pytest.fail("collateral initialization")
    context = OperationContext(project_containers=[flare], target_containers=[flare], target_services=("flare",))
    with manager.lifecycle.notify_start(context, prepare=False, generated_only=(flare,) if collateral else ()):
        assert sorted(calls) == [(name, manager.uid + 10000, False, {"group": manager.gid + 10000})
                                 for name in ("apps.yml", "bookmarks.yml")]
        flare.on_check(context)
    assert all(hook.source == "builtin" for hook in flare.hooks.iter_phase(HookPhase.BEFORE_START)
               if hook.name == "chown")
