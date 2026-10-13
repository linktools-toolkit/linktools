#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated inputs and dedicated application paths retain narrow access contracts."""
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr.artifacts import stage_files, sha256_of
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.process import RuntimeProcessFactory
from linktools.cntr.runtime import process as process_module


@pytest.fixture
def file_owner(tmp_path):
    return SimpleNamespace(
        name="app", manager=SimpleNamespace(data_path=tmp_path,
            artifact_index=SimpleNamespace(record=lambda entries: None)),
        get_app_path=lambda *parts: tmp_path.joinpath("app", *parts),
        runtime=SimpleNamespace(chmod=lambda path, mode: os.chmod(path, mode)),
    )


@pytest.mark.parametrize("mask", [0o022, 0o077])
@pytest.mark.parametrize("mode", [0o600, 0o640])
def test_prepared_directories_ignore_umask_without_relaxing_files(file_owner, mask, mode):
    previous = os.umask(mask)
    try:
        root = stage_files(file_owner, {"a/b/config": "private"}, mode=mode)
    finally:
        os.umask(previous)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o755
               for path in (root, root / "a", root / "a/b"))
    assert stat.S_IMODE((root / "a/b/config").stat().st_mode) == mode


def test_directory_policy_creates_new_tree_without_changing_legacy_generation(file_owner):
    files = {"nested/config": "private"}
    payload = json.dumps([0o600, None, sorted(files.items())], ensure_ascii=True, separators=(",", ":"))
    legacy = file_owner.get_app_path("generated", sha256_of(payload))
    (legacy / "nested").mkdir(parents=True)
    (legacy / "nested/config").write_text("private")
    legacy.chmod(0o700)
    (legacy / "nested").chmod(0o700)
    (legacy / "nested/config").chmod(0o600)
    current = legacy.parent / "current"
    current.symlink_to(legacy.name)
    candidate = stage_files(file_owner, files)
    assert candidate != legacy
    assert current.resolve() == legacy
    assert stat.S_IMODE(legacy.stat().st_mode) == 0o700
    assert stat.S_IMODE((legacy / "nested").stat().st_mode) == 0o700
    assert (legacy / "nested/config").read_text() == "private"


@pytest.mark.parametrize("change", ["root", "nested", "extra", "symlink"])
def test_reuse_rejects_directory_drift_without_repairing_active_tree(file_owner, change):
    files = {"nested/config": "private"}
    root = stage_files(file_owner, files)
    if change in ("root", "nested"):
        (root if change == "root" else root / "nested").chmod(0o700)
    elif change == "extra":
        (root / "extra").mkdir()
    else:
        (root / "extra").symlink_to(root / "nested", target_is_directory=True)
    with pytest.raises(ContainerError, match="modified"):
        stage_files(file_owner, files)
    if change in ("root", "nested"):
        assert stat.S_IMODE((root if change == "root" else root / "nested").stat().st_mode) == 0o700
    else:
        assert (root / "extra").exists()


@pytest.mark.parametrize("membership", ["primary", "supplementary", "absent"])
@pytest.mark.parametrize("owns_file", [True, False])
def test_prepared_group_change_uses_privilege_only_when_needed(file_owner, monkeypatch, membership, owns_file):
    target = os.getgid() + 10000
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() if owns_file else os.getuid() + 1)
    monkeypatch.setattr(os, "getegid", lambda: target if membership == "primary" else os.getgid())
    monkeypatch.setattr(os, "getgroups", lambda: [target] if membership == "supplementary" else [])
    calls = []
    file_owner.runtime.create_process = lambda *args, **kwargs: SimpleNamespace(
        check_call=lambda: calls.append((args, kwargs)))
    stage_files(file_owner, {"config": "private"}, mode=0o640, group=target)
    assert len(calls) == 1
    assert calls[0][0][0:2] == ("chgrp", str(target))
    assert calls[0][1]["privilege"] is not (owns_file and membership != "absent")


def test_unchanged_group_does_not_run_chgrp(file_owner):
    file_owner.runtime.create_process = lambda *args, **kwargs: pytest.fail("unnecessary chgrp")
    stage_files(file_owner, {"config": "private"}, mode=0o640, group=os.getgid())


@pytest.fixture
def ownership_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(process_module.shutil, "which", lambda name: "/usr/bin/" + name)
    runtime = RuntimeProcessFactory(SimpleNamespace(system="linux",
        env_config=SimpleNamespace(cast=lambda value, **kwargs: value),
        logger=SimpleNamespace(debug=lambda message: None, warning=lambda message: None)))
    calls = []
    runtime.create_process = lambda *args, **kwargs: SimpleNamespace(
        check_call=lambda: calls.append((args, kwargs)))
    path = tmp_path / "owned"
    path.mkdir()
    return runtime, calls, path


@pytest.mark.parametrize("numeric_string", [False, True])
def test_numeric_ownership_needs_no_host_account(ownership_runtime, monkeypatch, numeric_string):
    runtime, calls, path = ownership_runtime
    monkeypatch.setattr(process_module, "get_uid", lambda user: pytest.fail("host user lookup"))
    monkeypatch.setattr(process_module, "get_gid", lambda user: pytest.fail("host group lookup"))
    uid, gid = ("98765", "87654") if numeric_string else (98765, 87654)
    runtime.chown(path, uid, True, group=gid)
    assert calls[0][0] == ("chown", "-h", "-R", "98765:87654", str(path))


def test_named_user_keeps_primary_group_and_positional_recursive(ownership_runtime, monkeypatch):
    runtime, calls, path = ownership_runtime
    monkeypatch.setattr(process_module, "get_uid", lambda user: 1234)
    monkeypatch.setattr(process_module, "get_gid", lambda user: 4321)
    runtime.chown(path, "configured-user", True)
    assert calls[0][0] == ("chown", "-h", "-R", "1234:4321", str(path))


def test_numeric_user_without_group_preserves_existing_group(ownership_runtime):
    runtime, calls, path = ownership_runtime
    runtime.chown(path, 98765)
    assert calls[0][0] == ("chown", "98765", str(path))
    runtime.chown(path, 98765, True)
    assert calls[1][0] == ("chown", "-h", "-R", "98765", str(path))


@pytest.mark.parametrize("allowed", [True, False])
def test_numeric_group_change_checks_effective_memberships(ownership_runtime, monkeypatch, allowed):
    runtime, calls, path = ownership_runtime
    uid = path.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: uid)
    monkeypatch.setattr(os, "getegid", lambda: 11111)
    monkeypatch.setattr(os, "getgroups", lambda: [22222] if allowed else [])
    runtime.chown(path, uid, group=22222)
    assert calls[0][1]["privilege"] is not (allowed or uid == 0)


@pytest.mark.parametrize("uid", [0, 98765])
def test_template_ownership_accepts_lazy_numeric_ids_and_root(fresh_manager, tmp_path, monkeypatch, uid):
    owner = fresh_manager.containers["flare"]
    fresh_manager.env_config.set("DOCKER_UID", uid)
    fresh_manager.env_config.set("DOCKER_GID", 87654)
    hooks = []
    monkeypatch.setattr(owner, "add_start_hook", lambda key, hook, **kwargs: hooks.append(hook))
    calls = []
    monkeypatch.setattr(owner.runtime, "chown", lambda path, user, **kwargs: calls.append(
        (str(path), str(user), str(kwargs["group"]), kwargs["recursive"])))
    template = tmp_path / "compose.yml"
    template.write_text('{{ "/private/home" | chown(DOCKER_UID, True, group=DOCKER_GID) }}')
    assert owner.render_template(template) == "/private/home"
    assert len(hooks) == 1
    hooks[0]()
    assert calls == [("/private/home", str(uid), "87654", True)]


@pytest.mark.parametrize("existing", [False, True])
def test_flare_uses_runtime_ids_and_only_repairs_known_existing_paths(fresh_manager, tmp_path, monkeypatch, existing):
    owner = fresh_manager.containers["flare"]
    fresh_manager.env_config.set("DOCKER_UID", 98765)
    fresh_manager.env_config.set("DOCKER_GID", 87654)
    app = owner.get_app_path("runtime-app")
    legacy = owner.get_app_path("app")
    source = app if existing else legacy
    source.mkdir(parents=True)
    config = source / "config.yml"
    config.write_text("title: Keep my settings\n")
    config.chmod(0o400)
    assets = source / "assets"
    assets.mkdir()
    custom = assets / "private.txt"
    custom.write_text("user asset")
    custom.chmod(0o400)
    external = tmp_path / "external"
    external.write_text("untouched")
    external.chmod(0o400)
    (source / "external-link").symlink_to(external)
    before = (custom.stat(), external.stat())
    calls = []
    monkeypatch.setattr(owner.runtime, "chown", lambda path, user, recursive=False, **kwargs:
                        calls.append((Path(path), user, kwargs["group"], recursive)))
    monkeypatch.setattr(owner.runtime, "chmod", lambda path, mode, **kwargs: os.chmod(path, mode))
    written = []
    owner.on_starting(SimpleNamespace(write_files=lambda container, files, **kwargs: written.append((files, kwargs))))
    assert (app / "config.yml").read_text() == "title: Keep my settings\n"
    assert stat.S_IMODE((app / "config.yml").stat().st_mode) == 0o600
    assert stat.S_IMODE(app.stat().st_mode) == 0o750
    assert written[0][1] == {"mode": 0o640, "group": 87654}
    assert set(written[0][0]) == {"apps.yml", "bookmarks.yml"}
    assert all(user == 98765 and group == 87654 for _, user, group, _ in calls)
    if existing:
        assert {(path, recursive) for path, _, _, recursive in calls} == {(app, False), (app / "config.yml", False)}
        assert custom.stat() == before[0]
    else:
        assert len(calls) == 1 and calls[0][3] is True
        assert calls[0][0].name.startswith(".flare-app-")
        assert (legacy / "config.yml").read_text() == "title: Keep my settings\n"
        assert stat.S_IMODE((legacy / "config.yml").stat().st_mode) == 0o400
    assert external.stat() == before[1]
    assert (app / "external-link").is_symlink()


@pytest.mark.parametrize("link_target", ["root", "config"])
def test_flare_refuses_to_change_ownership_through_special_path_links(fresh_manager, tmp_path, monkeypatch, link_target):
    owner = fresh_manager.containers["flare"]
    app = owner.get_app_path("runtime-app")
    external = tmp_path / "outside"
    external.mkdir()
    config = external / "config.yml"
    config.write_text("untouched")
    config.chmod(0o400)
    if link_target == "root":
        app.parent.mkdir(parents=True, exist_ok=True)
        app.symlink_to(external, target_is_directory=True)
    else:
        app.mkdir(parents=True)
        (app / "config.yml").symlink_to(config)
    monkeypatch.setattr(owner.runtime, "chown", lambda *args, **kwargs: pytest.fail("must not chown"))
    monkeypatch.setattr(owner.runtime, "chmod", lambda *args, **kwargs: pytest.fail("must not chmod"))
    with pytest.raises(ContainerError, match="must"):
        owner.on_starting(SimpleNamespace(write_files=lambda *args, **kwargs: pytest.fail("must not publish")))
    assert config.read_text() == "untouched"
    assert stat.S_IMODE(config.stat().st_mode) == 0o400



def test_prepared_directory_policy_does_not_depend_on_runtime_chmod(file_owner):
    file_owner.runtime.chmod = lambda path, mode: None
    previous = os.umask(0o077)
    try:
        root = stage_files(file_owner, {"nested/config": "private"})
        assert stage_files(file_owner, {"nested/config": "private"}) == root
    finally:
        os.umask(previous)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "nested").stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "nested/config").stat().st_mode) == 0o600



def test_recursive_ownership_does_not_touch_external_symlink_targets(ownership_runtime, tmp_path):
    runtime, calls, root = ownership_runtime
    external = tmp_path / "outside.txt"
    external.write_text("private outside data")
    external.chmod(0o400)
    (root / "external-link").symlink_to(external)
    before = external.stat()

    def create_process(*args, **kwargs):
        assert kwargs["privilege"] is False
        return SimpleNamespace(check_call=lambda: subprocess.run(args, check=True))

    runtime.create_process = create_process
    runtime.chown(root, os.getuid(), True, group=os.getgid())
    assert external.stat() == before
    assert external.read_text() == "private outside data"


@pytest.mark.parametrize("dangling", [False, True])
def test_prepared_tree_root_link_is_rejected_without_following_it(file_owner, tmp_path, dangling):
    files = {"nested/config": "private"}
    root = stage_files(file_owner, files)
    retained = root.with_name("retained")
    root.rename(retained)
    root.symlink_to(tmp_path / "missing" if dangling else retained, target_is_directory=True)
    with pytest.raises(ContainerError, match="modified"):
        stage_files(file_owner, files)
    assert root.is_symlink()
    assert (retained / "nested/config").read_text() == "private"
