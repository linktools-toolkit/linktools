# -*- coding: utf-8 -*-
"""Tests for the Download module (spec §9).

Standalone infrastructure (PR 09); not yet wired into environ.downloads and does
not replace UrlFile -- that migration is a follow-up.
"""
import hashlib
import http.server
import os
import socketserver
import threading

import pytest

from linktools._download import (
    DownloadManager,
    DownloadRequest,
    FileTransport,
    HashValidator,
    SizeValidator,
    CompositeValidator,
    DownloadError,
)
from linktools._cache_store import CacheStore
from linktools.core._locks import LockManager
from linktools.types import MISSING


@pytest.fixture
def manager(tmp_path):
    environ = type("E", (), {
        "locks": LockManager(tmp_path / "locks"),
        "cache": CacheStore(tmp_path / "cache.db"),
    })()
    return DownloadManager(environ)


# --------------------------------------------------------------------------- #
# §9.7 validators
# --------------------------------------------------------------------------- #

def test_size_validator_passes_and_fails(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"12345")
    SizeValidator(5).validate(f)
    with pytest.raises(DownloadError):
        SizeValidator(6).validate(f)


def test_hash_validator_passes_and_fails(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    HashValidator(digest).validate(f)
    with pytest.raises(DownloadError):
        HashValidator("0" * 64).validate(f)


def test_composite_validator(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    CompositeValidator([SizeValidator(5), HashValidator(digest)]).validate(f)
    with pytest.raises(DownloadError):
        CompositeValidator([SizeValidator(99)]).validate(f)


# --------------------------------------------------------------------------- #
# §9.4 atomic landing + reuse-valid + lock, via FileTransport
# --------------------------------------------------------------------------- #

def test_file_transport_downloads_atomically(manager, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload-body")
    req = DownloadRequest(url=str(src), destination=tmp_path / "out" / "dst.bin")
    result = manager.download(req, transport=FileTransport())
    assert result.path.read_bytes() == b"payload-body"
    assert result.from_cache is False
    # no stray .part left beside the destination
    assert not list((tmp_path / "out").glob("*.part"))


def test_reuses_existing_valid_file(manager, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dst = tmp_path / "dst.bin"
    req = DownloadRequest(url=str(src), destination=dst, sha256=hashlib.sha256(b"abc").hexdigest())
    first = manager.download(req, transport=FileTransport())
    second = manager.download(req, transport=FileTransport())
    assert first.from_cache is False
    assert second.from_cache is True  # existing file validated, no re-download


def test_redownloads_when_hash_mismatch(manager, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"good")
    dst = tmp_path / "dst.bin"
    dst.write_bytes(b"stale-corrupt")  # pre-existing bad file
    req = DownloadRequest(url=str(src), destination=dst, sha256=hashlib.sha256(b"good").hexdigest())
    result = manager.download(req, transport=FileTransport())
    assert dst.read_bytes() == b"good"  # replaced
    assert result.from_cache is False


def test_hash_mismatch_after_download_raises_and_leaves_no_dst(manager, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"actual-content")
    dst = tmp_path / "dst.bin"
    req = DownloadRequest(url=str(src), destination=dst, sha256=hashlib.sha256(b"different").hexdigest())
    with pytest.raises(DownloadError):
        manager.download(req, transport=FileTransport())
    # destination must not exist (no half-installed file exposed)
    assert not dst.exists()
    assert not list(dst.parent.glob("*.part"))


# --------------------------------------------------------------------------- #
# §9.9 resume metadata lands in the cache store
# --------------------------------------------------------------------------- #

def test_resume_metadata_stored_in_cache(manager, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"data")
    req = DownloadRequest(url=str(src), destination=tmp_path / "dst.bin")
    manager.download(req, transport=FileTransport())
    ns = manager._environ.cache.namespace("download:resume")
    # at least one resume-metadata entry exists, keyed by the source
    assert len(ns.keys()) >= 1


# --------------------------------------------------------------------------- #
# HttpTransport basic fetch via a local server (200 path)
# --------------------------------------------------------------------------- #

def _serve(directory, handle):
    handler = type("H", (http.server.SimpleHTTPRequestHandler,), {
        "__init__": lambda self, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
            self, *a, directory=str(directory), **k),
        "log_message": lambda self, *a: None,
    })
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    handle["server"] = httpd
    handle["port"] = port
    httpd.serve_forever()


def test_http_transport_200(manager, tmp_path):
    body = b"http-body-" + bytes(range(256))
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "file.bin").write_bytes(body)
    handle = {}
    t = threading.Thread(target=_serve, args=(tmp_path / "srv", handle))
    t.daemon = True
    t.start()
    # wait for the server to bind
    import time as _t
    while "port" not in handle:
        _t.sleep(0.01)
    try:
        from linktools._download import HttpTransport
        url = "http://127.0.0.1:%d/file.bin" % handle["port"]
        req = DownloadRequest(url=url, destination=tmp_path / "dst.bin",
                              sha256=hashlib.sha256(body).hexdigest())
        result = manager.download(req, transport=HttpTransport())
        assert result.path.read_bytes() == body
    finally:
        handle["server"].shutdown()


def test_http_gzip_and_content_disposition(manager, tmp_path):
    import gzip

    raw = b"plain-body-" + bytes(range(256))
    gz = gzip.compress(raw)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Disposition", 'attachment; filename="real-name.bin"')
            self.send_header("Content-Length", str(len(gz)))
            self.end_headers()
            self.wfile.write(gz)

        def log_message(self, *a):
            return None

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    import threading as _t
    th = _t.Thread(target=httpd.serve_forever)
    th.daemon = True
    th.start()
    try:
        from linktools._download import HttpTransport
        url = "http://127.0.0.1:%d/x" % port
        req = DownloadRequest(url=url, destination=tmp_path / "dst.bin",
                              sha256=hashlib.sha256(raw).hexdigest())
        result = manager.download(req, transport=HttpTransport())
        # gzip was decompressed -> sha256 of the PLAIN body matches.
        assert result.path.read_bytes() == raw
        # Content-Disposition filename captured in resume metadata.
        meta = manager._environ.cache.namespace("download:resume").get(req.lock_key)
        assert meta.get("filename") == "real-name.bin"
    finally:
        httpd.shutdown()
