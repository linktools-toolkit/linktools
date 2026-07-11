#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic artifact writer and Generated Artifact Index."""
import json
import os

import pytest

from linktools.cntr.artifacts import ArtifactIndex, atomic_write_text_if_changed, sha256_of


def test_first_write_creates_file_and_returns_true(tmp_path):
    target = tmp_path / "out.txt"
    changed = atomic_write_text_if_changed(target, "hello")
    assert changed is True
    assert target.read_text() == "hello"


def test_unchanged_content_is_a_noop_and_returns_false(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text_if_changed(target, "hello")
    before_mtime = target.stat().st_mtime_ns

    changed = atomic_write_text_if_changed(target, "hello")

    assert changed is False
    assert target.stat().st_mtime_ns == before_mtime
    assert target.read_text() == "hello"


def test_changed_content_is_written_and_returns_true(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text_if_changed(target, "hello")
    changed = atomic_write_text_if_changed(target, "world")
    assert changed is True
    assert target.read_text() == "world"


def test_no_temp_file_left_behind_after_write(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text_if_changed(target, "hello")
    atomic_write_text_if_changed(target, "world")
    remaining = os.listdir(tmp_path)
    assert remaining == ["out.txt"]


def test_no_temp_file_left_behind_on_failure(tmp_path, monkeypatch):
    import linktools.utils as utils_module

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(utils_module, "atomic_write", fail)
    target = tmp_path / "out.txt"
    with pytest.raises(OSError):
        atomic_write_text_if_changed(target, "hello")
    assert os.listdir(tmp_path) == []


def test_existing_file_permissions_are_preserved_across_a_rewrite(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text_if_changed(target, "hello")
    os.chmod(target, 0o644)
    assert (os.stat(target).st_mode & 0o777) == 0o644

    changed = atomic_write_text_if_changed(target, "world")

    assert changed is True
    assert (os.stat(target).st_mode & 0o777) == 0o644
    assert target.read_text() == "world"


def test_encoding_round_trips_non_ascii(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text_if_changed(target, "héllo wörld")
    assert target.read_text(encoding="utf-8") == "héllo wörld"


# -- ArtifactIndex ------------------------------------------------------------

class _FakeManager:
    """Bare manager stand-in exposing only what ArtifactIndex needs, so
    these tests aren't coupled to whatever prepare_installed_containers()
    happens to write as a side effect on a real fresh_manager."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.project_name = "test-project"


def test_index_is_canonical_json_with_trailing_newline(tmp_path):
    index = ArtifactIndex(_FakeManager(tmp_path))
    index.record({"compose/nginx.yml": dict(kind="compose", container="nginx", sha256="abc", source=None)})

    with open(index.path, "r", encoding="utf-8") as f:
        content = f.read()

    assert content.endswith("\n")
    data = json.loads(content)
    assert data["schema_version"] == 1
    assert data["project"] == "test-project"
    # sort_keys=True -> re-serializing must byte-for-byte match the file.
    assert json.dumps(data, sort_keys=True, indent=2) + "\n" == content


def test_index_merges_and_preserves_unrelated_entries(tmp_path):
    index = ArtifactIndex(_FakeManager(tmp_path))
    index.record({"compose/a.yml": dict(kind="compose", container="a", sha256="1", source=None)})
    index.record({"compose/b.yml": dict(kind="compose", container="b", sha256="2", source=None)})

    artifacts = index.load()
    assert set(artifacts.keys()) == {"compose/a.yml", "compose/b.yml"}


def test_index_does_not_record_secrets_or_config_values(tmp_path):
    index = ArtifactIndex(_FakeManager(tmp_path))
    index.record({"compose/nginx.yml": dict(kind="compose", container="nginx", sha256="abc", source=None)})
    with open(index.path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "password" not in content.lower()
    entry = index.load()["compose/nginx.yml"]
    assert set(entry.keys()) <= {"kind", "container", "sha256", "source", "repository_url", "repository_revision"}


def test_sha256_of_is_stable_and_content_sensitive():
    assert sha256_of("a") == sha256_of("a")
    assert sha256_of("a") != sha256_of("b")


def test_load_returns_empty_dict_when_index_missing(tmp_path):
    assert ArtifactIndex(_FakeManager(tmp_path)).load() == {}


def test_load_returns_empty_dict_on_corrupt_index(tmp_path):
    index = ArtifactIndex(_FakeManager(tmp_path))
    os.makedirs(os.path.dirname(index.path), exist_ok=True)
    with open(index.path, "w", encoding="utf-8") as f:
        f.write("not json")
    assert index.load() == {}


# -- Wired into compose/Dockerfile generation ---------------------------------

def test_get_docker_compose_file_records_compose_artifact(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_docker_compose_file()
    artifacts = fresh_manager.artifact_index.load()
    assert "compose/nginx.yml" in artifacts
    assert artifacts["compose/nginx.yml"]["container"] == "nginx"
    assert artifacts["compose/nginx.yml"]["kind"] == "compose"
    assert len(artifacts["compose/nginx.yml"]["sha256"]) == 64


def test_prepare_installed_containers_records_dockerfile_artifact_as_a_side_effect(fresh_manager):
    # Rendering a container's compose (to auto-fill build.dockerfile)
    # resolves get_docker_file_path() as a side effect, which is where the
    # Dockerfile actually gets written -- exercised simply by accessing
    # docker_compose during prepare.
    artifacts = fresh_manager.artifact_index.load()
    assert "dockerfile/nginx.Dockerfile" in artifacts
    assert artifacts["dockerfile/nginx.Dockerfile"]["kind"] == "dockerfile"


def test_repo_backed_container_records_repository_url_and_revision(fresh_manager, tmp_path, monkeypatch):
    """The Generated Artifact Index's own canonical example entry includes both repository_url
    and repository_revision for a git-backed container -- only the former
    was ever actually recorded."""
    from linktools.cntr.container import BaseContainer
    from linktools.cntr.repo.context import RepositoryConfigContext
    import linktools.git as git_module

    class _FakeGitRepository:
        def __init__(self, environ, repo_path):
            pass

        def head_sha(self):
            return "cafef00d"

        @classmethod
        def open_if_valid(cls, environ, repo_path):
            return cls(environ, repo_path)

    monkeypatch.setattr(git_module, "GitRepository", _FakeGitRepository)

    (tmp_path / "docker-compose.yml").write_text("services:\n  app:\n    image: x:1\n")
    container = BaseContainer(fresh_manager, tmp_path, name="999-repo-backed")
    container._repository_context = RepositoryConfigContext(
        root_path=str(tmp_path), file_config=None, config=fresh_manager.env_config,
        url="https://example.invalid/repo.git", builtin=False,
    )

    container.get_docker_compose_file()

    entry = fresh_manager.artifact_index.load()["compose/repo-backed.yml"]
    assert entry["repository_url"] == "https://example.invalid/repo.git"
    assert entry["repository_revision"] == "cafef00d"


def test_non_git_local_repo_container_has_no_repository_revision(fresh_manager, tmp_path):
    from linktools.cntr.container import BaseContainer
    from linktools.cntr.repo.context import RepositoryConfigContext

    (tmp_path / "docker-compose.yml").write_text("services:\n  app:\n    image: x:1\n")
    container = BaseContainer(fresh_manager, tmp_path, name="999-local-repo")
    container._repository_context = RepositoryConfigContext(
        root_path=str(tmp_path), file_config=None, config=fresh_manager.env_config,
        url=str(tmp_path), builtin=False,
    )

    container.get_docker_compose_file()

    entry = fresh_manager.artifact_index.load()["compose/local-repo.yml"]
    assert entry["repository_url"] == str(tmp_path)
    assert "repository_revision" not in entry


def test_regenerating_unchanged_compose_does_not_touch_file_mtime(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    compose_path = nginx.get_docker_compose_file()
    before = os.stat(compose_path).st_mtime_ns

    # Re-render from a fresh container instance over the same data dir --
    # deterministic config means byte-identical output.
    fresh_manager.__dict__.pop("containers", None)
    nginx_again = fresh_manager.containers["nginx"]
    nginx_again.get_docker_compose_file()

    after = os.stat(compose_path).st_mtime_ns
    assert after == before
