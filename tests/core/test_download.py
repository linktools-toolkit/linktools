# -*- coding: utf-8 -*-
"""Tests for the Download module (spec §9)."""
import hashlib
import http.server
import os
import socketserver
import threading

import pytest

from linktools.core import (
    DownloadManager,
    DownloadRequest,
    FileTransport,
    HashValidator,
    SizeValidator,
    CompositeValidator,
)
from linktools.errors import DownloadError
from linktools.core import CacheStore
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
        from linktools.core import HttpTransport
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
        from linktools.core import HttpTransport
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


# --------------------------------------------------------------------------- #
# PR-3: resume / 416 state machine (spec §5.7)
# --------------------------------------------------------------------------- #

class _RangeServer:
    """Local HTTP server with a controllable resume/416 behaviour.

    ``mode`` selects how a ranged (Range: bytes=N-) request is answered; a
    request without Range always gets the full body as 200.
      good_206       -> 206 with a correct Content-Range (append)
      missing_cr     -> 206 but no Content-Range header (restart)
      bad_start      -> 206 whose Content-Range start != N (restart)
      416            -> 416 Range Not Satisfiable
      ignore_range   -> 200 full body, ignoring the Range header
    """

    def __init__(self, body, mode):
        self.body = body
        self.mode = mode
        self.range_requests = []

        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, code, payload, content_range=None):
                self.send_response(code)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                if content_range is not None:
                    self.send_header("Content-Range", content_range)
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                range_hdr = self.headers.get("Range")
                server.range_requests.append(range_hdr)
                total = len(server.body)
                if range_hdr and server.mode != "ignore_range":
                    n = int(range_hdr.split("=")[1].split("-")[0])
                    if server.mode == "416":
                        self.send_response(416); self.end_headers(); return
                    if server.mode == "missing_cr":
                        self._send(206, server.body[n:]); return
                    if server.mode == "bad_start":
                        cr = "bytes %d-%d/%d" % (n + 10, total - 1, total)
                        self._send(206, server.body[n:], content_range=cr); return
                    # good_206
                    cr = "bytes %d-%d/%d" % (n, total - 1, total)
                    self._send(206, server.body[n:], content_range=cr); return
                # no Range (or ignored) -> full body
                self._send(200, server.body)

            def log_message(self, *a):
                return None

        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    def url(self, path="/file.bin"):
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def shutdown(self):
        self._httpd.shutdown()


def _download_with_part(dst, part_bytes):
    """Pre-seed a .part file alongside the destination and return its path."""
    part = dst.parent / (dst.name + ".part")
    part.write_bytes(part_bytes)
    return part


def test_resume_206_good_content_range_appends(manager, tmp_path):
    body = b"X" * 4096
    srv = _RangeServer(body, "good_206")
    try:
        dst = tmp_path / "dst.bin"
        _download_with_part(dst, body[:1024])  # already have first 1024 bytes
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body
        assert srv.range_requests[0] == "bytes=1024-"  # resumed from 1024
    finally:
        srv.shutdown()


def test_resume_206_missing_content_range_restarts_full(manager, tmp_path):
    body = b"Y" * 2048
    srv = _RangeServer(body, "missing_cr")
    try:
        dst = tmp_path / "dst.bin"
        _download_with_part(dst, body[:512])  # stale/partial
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body
        # 1st ranged request returned a bad 206 -> transport restarted without Range.
        assert srv.range_requests[1] is None
    finally:
        srv.shutdown()


def test_resume_206_start_mismatch_restarts_full(manager, tmp_path):
    body = b"Z" * 2048
    srv = _RangeServer(body, "bad_start")
    try:
        dst = tmp_path / "dst.bin"
        _download_with_part(dst, body[:512])
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body
        assert srv.range_requests[1] is None  # restarted without Range
    finally:
        srv.shutdown()


def test_resume_200_overwrites_stale_part(manager, tmp_path):
    body = b"W" * 2048
    srv = _RangeServer(body, "ignore_range")  # server ignores Range -> 200
    try:
        dst = tmp_path / "dst.bin"
        _download_with_part(dst, b"stale-bytes")
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body  # overwritten, not appended
    finally:
        srv.shutdown()


def test_resume_416_complete_succeeds(manager, tmp_path):
    body = b"C" * 1024
    srv = _RangeServer(body, "416")
    try:
        dst = tmp_path / "dst.bin"
        # the .part already holds the full body; size hint in resume meta.
        _download_with_part(dst, body)
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        # Seed resume metadata (keyed by the request lock_key) with the
        # expected size so the transport can recognise the part as complete.
        manager._resume_namespace().set(req.lock_key, {"size": len(body)})
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body
    finally:
        srv.shutdown()


def test_resume_416_incomplete_restarts_full(manager, tmp_path):
    body = b"I" * 4096
    srv = _RangeServer(body, "416")
    try:
        dst = tmp_path / "dst.bin"
        _download_with_part(dst, body[:64])  # incomplete part
        req = DownloadRequest(url=srv.url(), destination=dst,
                              sha256=hashlib.sha256(body).hexdigest(), resume=True)
        manager._resume_namespace().set(req.lock_key, {"size": len(body)})
        result = manager.download(req, transport=_HttpTransport())
        assert result.path.read_bytes() == body
        # 1st ranged -> 416 (incomplete) -> discarded -> 2nd full request.
        assert srv.range_requests[1] is None
    finally:
        srv.shutdown()


def _HttpTransport():
    from linktools.core import HttpTransport
    return HttpTransport()


# --------------------------------------------------------------------------- #
# PR-3 (fix-plan §3.3): hash-mismatch retry-once / second-fail semantics
# --------------------------------------------------------------------------- #

class _ScriptedTransport:
    """Serves a scripted sequence of payloads so hash-retry can be exercised.

    Used instead of a real HTTP server: fetch N writes payloads[N] to ``part``
    (clamping to the last), so a [bad, good] sequence models a transient bad
    fetch that a retry fixes.
    """

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self._i = 0
        self.fetches = 0

    def fetch(self, request, part, on_progress=None, meta=None):
        from pathlib import Path
        payload = self._payloads[min(self._i, len(self._payloads) - 1)]
        self._i += 1
        self.fetches += 1
        Path(part).parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as f:
            f.write(payload)
        if meta is not None:
            meta["url"] = request.url


def test_hash_mismatch_retry_once_then_succeeds(manager, tmp_path):
    # First fetch lands bad content (hash mismatch); the retry lands good
    # content and the download succeeds.
    good = b"good-content"
    bad = b"bad-content"
    dst = tmp_path / "dst.bin"
    req = DownloadRequest(url="mem://x", destination=dst,
                          sha256=hashlib.sha256(good).hexdigest())
    transport = _ScriptedTransport([bad, good])
    result = manager.download(req, transport=transport)
    assert transport.fetches == 2          # exactly one retry
    assert result.path.read_bytes() == good


def test_second_hash_mismatch_fails(manager, tmp_path):
    # Both fetches are wrong -> retry once, then fail; no infinite loop, no dst.
    bad = b"bad-content"
    dst = tmp_path / "dst.bin"
    req = DownloadRequest(url="mem://x", destination=dst,
                          sha256=hashlib.sha256(b"different").hexdigest())
    transport = _ScriptedTransport([bad, bad])
    with pytest.raises(DownloadError):
        manager.download(req, transport=transport)
    assert transport.fetches == 2          # initial + one retry, not infinite
    assert not dst.exists()                # no half-installed file exposed
