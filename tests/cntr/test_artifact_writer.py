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

class _FakeEnviron:
    def __init__(self, data_path):
        from linktools.core._locks import LockManager
        self.locks = LockManager(data_path / "locks")


class _FakeManager:
    """Bare manager stand-in exposing only what ArtifactIndex needs, so
    these tests aren't coupled to whatever prepare_installed_containers()
    happens to write as a side effect on a real fresh_manager."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.project_name = "test-project"
        self.environ = _FakeEnviron(data_path)


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


def test_load_fails_closed_on_corrupt_index(tmp_path):
    """review P1-12: a corrupt index must not be silently treated as an
    empty one -- that would make record() discard every prior entry and
    Plan/Doctor treat every real artifact as newly "added"."""
    from linktools.cntr.artifacts import ArtifactIndexError

    index = ArtifactIndex(_FakeManager(tmp_path))
    os.makedirs(os.path.dirname(index.path), exist_ok=True)
    with open(index.path, "w", encoding="utf-8") as f:
        f.write("not json")
    with pytest.raises(ArtifactIndexError):
        index.load()


@pytest.mark.parametrize("payload", [
    "[]",
    '{"schema_version": 1, "project": "x", "artifacts": []}',
    '{"schema_version": 1, "project": "x", "artifacts": {"a": "not-an-object"}}',
    '{"schema_version": 999, "project": "x", "artifacts": {}}',
    '{"project": "x", "artifacts": {}}',
    '{"schema_version": 1, "project": 123, "artifacts": {}}',
])
def test_load_fails_closed_on_structurally_invalid_index(tmp_path, payload):
    from linktools.cntr.artifacts import ArtifactIndexError

    index = ArtifactIndex(_FakeManager(tmp_path))
    os.makedirs(os.path.dirname(index.path), exist_ok=True)
    with open(index.path, "w", encoding="utf-8") as f:
        f.write(payload)
    with pytest.raises(ArtifactIndexError):
        index.load()


# -- Wired into compose/Dockerfile generation ---------------------------------

def test_get_docker_compose_file_records_compose_artifact(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_docker_compose_file()
    artifacts = fresh_manager.artifact_index.load()
    assert "compose/nginx.yml" in artifacts
    assert artifacts["compose/nginx.yml"]["container"] == "nginx"
    assert artifacts["compose/nginx.yml"]["kind"] == "compose"
    assert len(artifacts["compose/nginx.yml"]["sha256"]) == 64


def test_prepare_installed_containers_alone_does_not_write_dockerfile(fresh_manager):
    """review P1-11: merely preparing containers (accessing docker_compose,
    e.g. for config list/Plan/Doctor) must never write a Dockerfile as a
    side effect -- only real compose file generation for actual execution
    does (see the next test)."""
    artifacts = fresh_manager.artifact_index.load()
    assert "dockerfile/nginx.Dockerfile" not in artifacts


def test_real_compose_generation_writes_the_dockerfile(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.get_docker_compose_file()

    artifacts = fresh_manager.artifact_index.load()
    assert "dockerfile/nginx.Dockerfile" in artifacts
    assert artifacts["dockerfile/nginx.Dockerfile"]["kind"] == "dockerfile"


def test_repo_backed_container_records_repo_name_and_revision_not_url(fresh_manager, tmp_path, monkeypatch):
    """The Generated Artifact Index's canonical example entry includes
    repo_name/repo_id and repository_revision for a git-backed container --
    it must NEVER include repository_url (review P1-07): the add URL may
    embed a Git credential (`https://user:token@host/repo.git`), and the
    Artifact Index has no secret redaction of its own."""
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
    container.repo_context = RepositoryConfigContext(
        root_path=str(tmp_path), file_config=None,
        url="https://token@example.invalid/repo.git", builtin=False, repo_name="repo",
    )

    container.get_docker_compose_file()

    entry = fresh_manager.artifact_index.load()["compose/repo-backed.yml"]
    assert entry["repo_name"] == "repo"
    assert "repo_id" in entry
    assert entry["repository_revision"] == "cafef00d"
    assert "repository_url" not in entry
    assert "token" not in str(entry)


def test_non_git_local_repo_container_has_no_repository_revision(fresh_manager, tmp_path):
    from linktools.cntr.container import BaseContainer
    from linktools.cntr.repo.context import RepositoryConfigContext

    (tmp_path / "docker-compose.yml").write_text("services:\n  app:\n    image: x:1\n")
    container = BaseContainer(fresh_manager, tmp_path, name="999-local-repo")
    container.repo_context = RepositoryConfigContext(
        root_path=str(tmp_path), file_config=None,
        url=str(tmp_path), builtin=False, repo_name="local-repo",
    )

    container.get_docker_compose_file()

    entry = fresh_manager.artifact_index.load()["compose/local-repo.yml"]
    assert entry["repo_name"] == "local-repo"
    assert "repository_url" not in entry
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


@pytest.mark.parametrize("compose,docker_file", [
    ({"services": {"app": {"image": "example:1", "labels": {"label": "café"}}}}, "FROM scratch\n"),
    ({"services": {"app": {"image": "example:1"}}}, None),
    (None, "FROM scratch\n"),
    ({}, ""),
])
def test_candidates_match_written_content_without_creating_paths(tmp_path, compose, docker_file):
    from types import SimpleNamespace
    from linktools.cntr.artifacts import collect_candidates
    from linktools.cntr._container.compose import write_docker_compose_file, write_docker_file

    data_path = tmp_path / "not-created"
    manager = _FakeManager(data_path)
    manager.docker_compose_names = ("docker-compose.yml",)
    manager.artifact_index = ArtifactIndex(manager)
    container = SimpleNamespace(
        manager=manager, name="app", docker_compose=compose, docker_file=docker_file,
        repo_context=None, get_source_path=lambda name: tmp_path / "sources" / name,
    )

    candidates = collect_candidates(manager, (container,))

    assert not data_path.exists()
    compose_path = write_docker_compose_file(container)
    docker_file_path = write_docker_file(container)
    written = {str(path) for path in (compose_path, docker_file_path) if path is not None}
    assert set(candidates) == written
    for path, (kind, owner, content) in candidates.items():
        assert owner == "app"
        with open(path, encoding="utf-8") as stream:
            assert stream.read() == content
        entry = manager.artifact_index.load()[os.path.relpath(path, str(data_path))]
        assert entry["kind"] == kind
        assert entry["sha256"] == sha256_of(content)


def test_candidate_serializers_read_each_model_once(tmp_path):
    from linktools.cntr.artifacts import collect_candidates

    class Container:
        name = "app"
        manager = _FakeManager(tmp_path / "unused")
        compose_reads = 0
        docker_file_reads = 0

        @property
        def docker_compose(self):
            self.compose_reads += 1
            return {"services": {"app": {"image": "example:1"}}}

        @property
        def docker_file(self):
            self.docker_file_reads += 1
            return "FROM scratch\n"

    container = Container()
    assert len(collect_candidates(container.manager, (container,))) == 2
    assert container.compose_reads == 1
    assert container.docker_file_reads == 1


def test_invalid_compose_candidate_fails_before_creating_destination(tmp_path):
    from types import SimpleNamespace
    import yaml
    from linktools.cntr.artifacts import collect_candidates
    from linktools.cntr._container.compose import write_docker_compose_file

    manager = _FakeManager(tmp_path / "not-created")
    container = SimpleNamespace(
        manager=manager, name="app", docker_compose={"unsafe": object()}, docker_file=None,
    )
    for render in (lambda: collect_candidates(manager, (container,)),
                   lambda: write_docker_compose_file(container)):
        with pytest.raises(yaml.representer.RepresenterError):
            render()
        assert not manager.data_path.exists()
