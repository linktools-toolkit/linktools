#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr import SourceContainer
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.images import ImagePreparer, LocalImage

if TYPE_CHECKING:
    from typing import Type


class Source(SourceContainer):
    @property
    def _source_url(self):
        return "https://example.com/source.zip"

    @property
    def _source_path(self):
        return "pkg"

    def _handle_source_file(self, source, destination):
        with zipfile.ZipFile(source) as archive:
            archive.extractall(destination)


def make_archive(path, version):
    with zipfile.ZipFile(str(path), "w") as archive:
        archive.writestr("pkg/version.txt", version)


def case(tmp_path):
    source = tmp_path / "upstream.zip"
    make_archive(source, "first")
    requests = []

    class File:
        def save(self, directory, name):
            target = Path(directory) / name
            shutil.copyfile(str(source), str(target))
            return str(target)

    def get_file(url):
        requests.append(url)
        return File()

    manager = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args: None),
        app_path=tmp_path, environ=SimpleNamespace(get_url_file=get_file),
        image_preparer=SimpleNamespace(image_exists=lambda image: False))
    container = Source(manager, tmp_path, name="source")
    container.__dict__["services"] = {"web": {"image": "web:local", "build": {}}}
    return container, source, requests


def build_model(container):
    prep = ImagePreparer(container.manager)
    model = {"services": {"web": {"image": "web:local", "build": {
        "context": str(container.get_docker_context_path()), "dockerfile_inline": "FROM scratch"}}}}
    return prep, prep.with_build_revisions(model, (container,), ("web",))


def test_source_revision_refresh_and_no_network_when_stable(tmp_path):
    container, archive, calls = case(tmp_path)
    path = container.get_docker_context_path()
    assert not calls
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    assert len(calls) == 1 and (path / "version.txt").read_text() == "first"
    before = build_model(container)[1]
    make_archive(archive, "second")
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    assert len(calls) == 1 and build_model(container)[1] == before
    container._prepare_source(SimpleNamespace(refresh_services=frozenset(("web",))))
    assert len(calls) == 2
    assert (path / "version.txt").read_text() == "second"
    prep, after = build_model(container)
    assert after != before
    previous = before["services"]["web"]["build"]["labels"][prep.BUILD_LABEL]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:old", previous) for image in images}
    assert prep.plan(after).build == ("web",)


def test_missing_source_tree_restores_from_archive_without_network(tmp_path):
    container, _, calls = case(tmp_path)
    ctx = SimpleNamespace(refresh_services=frozenset())
    container._prepare_source(ctx)
    digest = container._source_digest()
    shutil.rmtree(str(container._source_root / "versions" / digest))
    container._prepare_source(ctx)
    assert len(calls) == 1
    container.prepare_build_context()
    assert len(calls) == 1
    assert (container.get_docker_context_path() / "version.txt").read_text() == "first"


def test_missing_cached_source_never_substitutes_moved_remote_tag(tmp_path):
    container, source, calls = case(tmp_path)
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    digest = container._source_digest()
    shutil.rmtree(str(container._source_root / "versions" / digest))
    (container._source_root / "archives" / (digest + ".in")).unlink()
    make_archive(source, "upstream moved")
    with pytest.raises(ContainerError, match="changed unexpectedly"):
        container.prepare_build_context()
    assert len(calls) == 2
    assert container._source_digest() == digest


def test_download_failure_preserves_existing_snapshot(tmp_path):
    container, source, calls = case(tmp_path)
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    digest = container._source_digest()

    def fail(url):
        raise OSError("download failed")
    container.manager.environ.get_url_file = fail
    with pytest.raises(OSError, match="download failed"):
        container._prepare_source(SimpleNamespace(refresh_services=frozenset(("web",))))
    assert container._source_digest() == digest
    assert (container.get_docker_context_path() / "version.txt").read_text() == "first"

def test_existing_image_never_fetches_uncached_source_during_plain_up(tmp_path):
    container, _, calls = case(tmp_path)
    container.manager.image_preparer.image_exists = lambda image: True
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    assert calls == []
    assert container._source_digest() is None
    prep, model = build_model(container)
    prep.inspect_images = lambda images: {image: LocalImage("sha256:existing", "old-label") for image in images}
    assert prep.plan(model).build == ()
    assert calls == []
    container._prepare_source(SimpleNamespace(refresh_services=frozenset(("web",))))
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, OSError])
@pytest.mark.parametrize("stage", ["tree", "archive"])
def test_interrupted_legacy_migration_never_publishes_partial_inputs(
        tmp_path: Path, monkeypatch: "pytest.MonkeyPatch", error_type: "Type[BaseException]", stage: str) -> None:
    import hashlib
    from linktools.utils import get_md5

    container, upstream, requests = case(tmp_path)
    name = get_md5(container._source_url)
    archive = container.get_app_path("source", name + ".in")
    tree = container.get_app_path("source", name + ".out")
    archive.parent.mkdir(parents=True)
    shutil.copy2(str(upstream), str(archive))
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "version.txt").write_text("first")
    original = archive.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    context = SimpleNamespace(refresh_services=frozenset())

    def interrupted_copy(source: str, destination: str) -> None:
        path = Path(destination)
        if stage == "tree":
            (path / "pkg").mkdir(parents=True)
            path = path / "pkg" / "partial.txt"
        path.write_text("partial")
        raise error_type("interrupted copy")

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copytree" if stage == "tree" else "copy2", interrupted_copy)
        with pytest.raises(error_type, match="interrupted copy"):
            container._prepare_source(context)

    assert container._source_digest() is None
    assert not list(container._source_root.glob(".migrate-*"))
    assert not (container._source_root / "archives" / (digest + ".in")).exists()
    if stage == "tree":
        assert not (container._source_root / "versions" / digest).exists()
    assert archive.read_bytes() == original
    assert (tree / "pkg" / "version.txt").read_text() == "first"

    container._prepare_source(context)

    assert requests == []
    assert container._source_digest() == digest
    assert (container.get_docker_context_path() / "version.txt").read_text() == "first"
    assert not (container.get_docker_context_path() / "partial.txt").exists()
    assert (container._source_root / "archives" / (digest + ".in")).read_bytes() == original


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, OSError])
def test_interrupted_archive_retention_preserves_current_source_and_retries(
        tmp_path: Path, monkeypatch: "pytest.MonkeyPatch", error_type: "Type[BaseException]") -> None:
    import hashlib
    import os

    container, upstream, requests = case(tmp_path)
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    previous = container._source_digest()
    make_archive(upstream, "second")
    expected_archive = upstream.read_bytes()
    digest = hashlib.sha256(expected_archive).hexdigest()
    retained = container._source_root / "archives" / (digest + ".in")
    context = SimpleNamespace(refresh_services=frozenset(("web",)))
    replace = os.replace

    def interrupted_replace(source: str, destination: str) -> None:
        if Path(destination) == retained:
            assert Path(source).read_bytes() == expected_archive
            raise error_type("interrupted archive publication")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted_replace)
        with pytest.raises(error_type, match="interrupted archive publication"):
            container._prepare_source(context)

    assert container._source_digest() == previous
    assert (container.get_docker_context_path() / "version.txt").read_text() == "first"
    assert not retained.exists()
    assert not list(container._source_root.glob(".source-*"))
    assert not list(container._source_root.glob(".extract-*"))
    container._prepare_source(context)
    assert len(requests) == 3
    assert container._source_digest() == digest
    assert (container.get_docker_context_path() / "version.txt").read_text() == "second"
    assert retained.read_bytes() == expected_archive
