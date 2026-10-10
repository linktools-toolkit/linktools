#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy certificate copies remain readable without relaxing private modes."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("system, uid, docker_type", [
    ("linux", 1000, "docker"),
    ("darwin", 1000, "docker"),
    ("linux", 1000, "docker-rootless"),
    ("linux", 0, "docker"),
])
@pytest.mark.parametrize("existing_backup", [False, True])
def test_legacy_private_keys_are_owned_before_reading(
        fresh_manager, monkeypatch, system, uid, docker_type, existing_backup):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_app_path().mkdir(parents=True, exist_ok=True)
    for name, value in (("system", system), ("uid", uid), ("gid", uid),
                        ("container_type", docker_type)):
        monkeypatch.setattr(type(fresh_manager), name, value)
    privileged_copy = uid != 0 and docker_type == "docker"
    pending_ownership = set()
    calls = []

    def seed(destination):
        destination.mkdir(parents=True, exist_ok=True)
        nested = destination / "private"
        nested.mkdir(mode=0o700)
        key = nested / "key.pem"
        key.write_bytes(b"synthetic-private-key")
        key.chmod(0o600)
        (destination / "key.pem").symlink_to("private/key.pem")
        if privileged_copy:
            pending_ownership.add(str(key))

    if existing_backup:
        for name in ("certs", "acme"):
            seed(nginx.get_app_path("migration-backup", name))

    def create_process(*args, **kwargs):
        calls.append((args, kwargs))

        def check_call():
            if args[:2] == ("docker", "cp"):
                assert kwargs["privilege"] == (docker_type == "docker")
                seed(Path(args[-1]))
            else:
                assert args == ("chown", "-R", "-h", "{}:{}".format(uid, uid),
                                str(nginx.get_app_path("migration-backup")))
                assert kwargs == {"privilege": True}
                pending_ownership.clear()
        return SimpleNamespace(check_call=check_call)

    monkeypatch.setattr(nginx.runtime, "create_process", create_process)
    copy = shutil.copy2

    def copy_private_file(source, destination):
        # Model root-owned 0600 files even when the test runner itself is root.
        if pending_ownership:
            raise PermissionError("Private backup files still belong to root")
        return copy(source, destination)

    monkeypatch.setattr(shutil, "copy2", copy_private_file)
    nginx._preserve_legacy_files()

    assert sum(args[0] == "chown" for args, _ in calls) == int(privileged_copy)
    assert sum(args[0] == "docker" for args, _ in calls) == (0 if existing_backup else 2)
    for name in ("certs", "acme"):
        key = nginx.get_app_path(name, "private", "key.pem")
        assert key.read_bytes() == b"synthetic-private-key"
        assert key.stat().st_mode & 0o777 == 0o600
        assert os.readlink(str(nginx.get_app_path(name, "key.pem"))) == "private/key.pem"
        backup_key = nginx.get_app_path("migration-backup", name, "private", "key.pem")
        assert backup_key.stat().st_mode & 0o777 == 0o600

    # A retry must not replace files already installed by migration or renewal.
    key.write_bytes(b"newer-private-key")
    nginx._preserve_legacy_files()
    assert key.read_bytes() == b"newer-private-key"
