#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr import SourceContainer
from linktools.cntr.errors import ContainerError


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
        app_path=tmp_path, environ=SimpleNamespace(get_url_file=get_file))
    container = Source(manager, tmp_path, name="source")
    container.__dict__["services"] = {"web": {"image": "web:local", "build": {}}}
    return container, source, requests


def test_source_revision_refresh_and_no_network_when_stable(tmp_path):
    container, archive, calls = case(tmp_path)
    path = container.get_docker_context_path()
    assert not calls
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    assert len(calls) == 1 and (path / "version.txt").read_text() == "first"
    before = container.get_build_revision("web")
    make_archive(archive, "second")
    container._prepare_source(SimpleNamespace(refresh_services=frozenset()))
    assert len(calls) == 1 and container.get_build_revision("web") == before
    container._prepare_source(SimpleNamespace(refresh_services=frozenset(("web",))))
    assert len(calls) == 2
    assert (path / "version.txt").read_text() == "second"
    assert container.get_build_revision("web") != before


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
