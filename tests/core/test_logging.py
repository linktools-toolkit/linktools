# -*- coding: utf-8 -*-
"""Tests for the LoggingManager redaction + context ( LOG-003,
 LOG-004).

The manager is the single place log records are scrubbed of secrets and
annotated with thread-local context. Business modules must never call
``logging.basicConfig``/``addHandler``/``setLevel`` themselves --
those go through the manager.
"""
import logging
import re
import threading

import pytest

from linktools.core._logging import LoggingManager


@pytest.fixture
def manager():
    return LoggingManager()


# --------------------------------------------------------------------------- #
# Secret redaction -- the built-in scrubbers must cover the categories
# listed in the spec without any registration.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, leak", [
    ("https://alice:s3cr3t@example.com/repo.git", "s3cr3t"),
    ("http://root:hunter2@host:8080/p", "hunter2"),
    ("Authorization: Bearer eyJhbGci.x.y", "eyJhbGci.x.y"),
    ("authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("Cookie: sessionid=abcdef0123", "abcdef0123"),
    ("Set-Cookie: token=xyz; Path=/", "xyz"),
    ("GET /?token=supersecret&name=ok", "supersecret"),
    ("?api_key=AKIA1234567890&dry=1", "AKIA1234567890"),
    ("x-access-token: deadbeef", "deadbeef"),
    ("password=p@ssw0rd", "p@ssw0rd"),
    ("sshpass -p letmein ssh host", "letmein"),
])
def test_builtin_redaction_masks_secrets(manager, raw, leak):
    out = manager.redact(raw)
    assert leak not in out, "leaked %r -> %r" % (leak, out)
    assert "***" in out


def test_register_secret_masks_literal(manager):
    manager.register_secret("a-very-private-key")
    out = manager.redact("config key=a-very-private-key done")
    assert "a-very-private-key" not in out
    assert "***" in out


def test_register_secret_ignores_empty_and_non_string(manager):
    manager.register_secret("")
    manager.register_secret(None)  # type: ignore[arg-type]
    # Empty/non-string secrets register nothing, so clean text is untouched.
    assert manager.redact("nothing changes here") == "nothing changes here"


def test_register_redactor_with_pattern(manager):
    manager.register_redactor(re.compile(r"API_KEY=\w+"))
    out = manager.redact("cfg API_KEY=ABC123 end")
    assert "ABC123" not in out


def test_redact_non_string_passthrough(manager):
    assert manager.redact(123) == 123
    assert manager.redact(None) is None


# --------------------------------------------------------------------------- #
# Adversarial regressions (found by redaction review, all fixed).
# --------------------------------------------------------------------------- #

def test_url_password_containing_at_is_fully_masked(manager):
    # RFC userinfo terminator is the LAST '@'; the password 'p@ss' must not
    # leak its tail ('ss').
    out = manager.redact("https://u:p@ss@host/x")
    assert "p@ss" not in out and "ss@" not in out
    assert "host" in out  # host survives


def test_multi_word_secret_keys_are_masked(manager):
    assert "AKIA1234567890" not in manager.redact("api key=AKIA1234567890")
    assert "topsecret" not in manager.redact("secret key=topsecret")
    assert "s3cr3t" not in manager.redact("client secret=s3cr3t")


def test_no_redos_on_long_input(manager):
    import time as _time
    start = _time.monotonic()
    manager.redact("a" * 100000)
    elapsed = _time.monotonic() - start
    # Bounded prefix keeps this linear; well under a second.
    assert elapsed < 1.0, "redact took %.2fs (possible ReDoS)" % elapsed


def test_format_string_logging_does_not_crash(manager):
    # logger.info("password=%s", pwd) must not raise TypeError after redaction
    # clobbers the %s placeholder.
    manager.install_filter()
    try:
        recs = _capture(manager, "lt.test.fmt", "password=%s", "letmein")
        assert recs
        rendered = recs[0].getMessage()
        assert "letmein" not in rendered
        assert "***" in rendered
    finally:
        manager.remove_filter()


def test_format_string_with_url_arg_redacts(manager):
    manager.install_filter()
    try:
        recs = _capture(manager, "lt.test.fmturl", "clone %s", "https://u:p@h/x")
        assert "https://***@h/x" in recs[0].getMessage()
    finally:
        manager.remove_filter()


def test_redact_does_not_touch_clean_text(manager):
    msg = "downloaded frida-server to /data/frida/fs"
    # a clean URL/path must survive (no false-positive clobbering of the path)
    out = manager.redact("https://github.com/frida/frida/releases")
    assert "github.com/frida/frida/releases" in out


# --------------------------------------------------------------------------- #
# Context -- thread-local, scoped via a context manager.
# --------------------------------------------------------------------------- #

def test_context_attaches_fields_and_pops_on_exit(manager):
    with manager.context(task_id="t1", tool="adb"):
        assert manager.current_context() == {"task_id": "t1", "tool": "adb"}
    assert manager.current_context() == {}


def test_context_is_thread_local(manager):
    barrier = threading.Event()
    seen = {}

    def worker():
        with manager.context(device="emulator-5554"):
            barrier.wait()
            seen["worker"] = manager.current_context()

    t = threading.Thread(target=worker)
    t.start()
    with manager.context(device="main-thread-device"):
        barrier.set()
        t.join()
    assert seen["worker"] == {"device": "emulator-5554"}
    # main thread's context did not leak into the worker
    assert seen["worker"].get("device") != "main-thread-device"


# --------------------------------------------------------------------------- #
# End-to-end: a record flowing through a logger is scrubbed + annotated.
# --------------------------------------------------------------------------- #

def _capture(manager, name, msg, *args):
    logger = logging.getLogger(name)
    recs = []

    class Cap(logging.Handler):
        def emit(self, record):
            recs.append(record)

    cap = Cap()
    logger.addHandler(cap)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info(msg, *args)
    finally:
        logger.removeHandler(cap)
    return recs


def test_records_are_redacted_end_to_end(manager):
    manager.install_filter()  # attach the redacting filter to the root logger
    try:
        recs = _capture(manager, "lt.test.redact", "clone %s", "https://alice:s3cr3t@example.com/x")
        assert recs
        rendered = recs[0].getMessage()
        assert "s3cr3t" not in rendered
    finally:
        manager.remove_filter()


def test_context_fields_reach_records(manager):
    manager.install_filter()
    try:
        with manager.context(task_id="abc"):
            recs = _capture(manager, "lt.test.ctx", "running")
        assert recs[0].task_id == "abc"  # type: ignore[attr-defined]
    finally:
        manager.remove_filter()


def test_install_filter_is_idempotent(manager):
    # The global environ may already have a redactor active (via get_logger);
    # verify this manager doesn't double-wrap and restores to the prior state.
    factory_before = logging.getLogRecordFactory()
    manager.install_filter()
    first = logging.getLogRecordFactory()
    manager.install_filter()  # must not double-wrap
    second = logging.getLogRecordFactory()
    assert first is second
    manager.remove_filter()
    assert logging.getLogRecordFactory() is factory_before
