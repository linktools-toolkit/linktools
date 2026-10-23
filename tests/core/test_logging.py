#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the LoggingManager redaction + context ( LOG-003,
 LOG-004).

The manager is the single place log records are scrubbed of secrets and
annotated with thread-local context. Business modules must never call
``logging.basicConfig``/``addHandler``/``setLevel`` themselves --
those go through the manager.
"""
import ast
import json
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
    value = {"password": "not-a-string"}
    assert manager.redact(value) is value


@pytest.mark.parametrize("raw, expected", [
    ('password="sensitive value" status=401 detail="try again"',
     'password="***" status=401 detail="try again"'),
    ("secret='sensitive value' status=500", "secret='***' status=500"),
    ('{"error":{"api_key":"sensitive value","status":401},"detail":"try again"}',
     '{"error":{"api_key":"***","status":401},"detail":"try again"}'),
    ("{'error': {'client_secret': 'sensitive value', 'status': 401}}",
     "{'error': {'client_secret': '***', 'status': 401}}"),
    ('{"token":["first",{"nested":"second"}],"status":401}',
     '{"token":"***","status":401}'),
    ('{"password":12345,"status":401}', '{"password":***,"status":401}'),
    ('[("Authorization", "Bearer private-value"), ("status", "401")]',
     '[("Authorization", "***"), ("status", "401")]'),
    ("[('X-API-Key', 'private-value'), ('status', '401')]",
     "[('X-API-Key', '***'), ('status', '401')]"),
    ("[(b'X-API-Key', b'private-value'), (b'status', b'401')]",
     "[(b'X-API-Key', b'***'), (b'status', b'401')]"),
    ('{"name":"Authorization","value":"Basic dXNlcjpwYXNz","status":401}',
     '{"name":"Authorization","value":"***","status":401}'),
    ("{'value': 'private-value', 'name': 'X-Api-Key', 'status': 401}",
     "{'value': '***', 'name': 'X-Api-Key', 'status': 401}"),
    ('https://u:p@ss@host/x?api%5Fkey=private%20value&status=401',
     'https://***@host/x?api%5Fkey=***&status=401'),
    ('https://u:p%2Fss%40private@host/x?token=private%26value%23tail&status=401',
     'https://***@host/x?token=***&status=401'),
    ('https%3A%2F%2Fu%3Ap%40ss%40host%2Fx%3Fapi%5Fkey%3Dprivate%26status%3D401',
     'https%3A%2F%2F***%40host%2Fx%3Fapi%5Fkey%3D***%26status%3D401'),
    ('?access_token=private-value&detail=retry', '?access_token=***&detail=retry'),
    ('?X%2DApi%2DKey=private-value&detail=retry', '?X%2DApi%2DKey=***&detail=retry'),
    ('request failed with Bearer private-value; status=401',
     'request failed with Bearer ***; status=401'),
    ('request failed with Basic dXNlcjpwYXNz; status=401',
     'request failed with Basic ***; status=401'),
    ("sshpass -p 'sensitive value' ssh host", "sshpass -p '***' ssh host"),
    ("sshpass -p sensitive ssh host", "sshpass -p *** ssh host"),
    ('password="""sensitive value""" status=401', 'password="""***""" status=401'),
])
def test_structured_secrets_preserve_surrounding_diagnostics(manager, raw, expected):
    assert manager.redact(raw) == expected
    assert manager.redact(expected) == expected


@pytest.mark.parametrize("layers", [0, 1, 2, 3])
def test_escaped_json_secrets_preserve_fields_and_escaped_quotes(manager, layers):
    secret = 'private-value with "quotes" and \\slashes'
    raw = json.dumps({"api_key": secret, "status": 401, "detail": "try again"})
    expected = json.dumps({"api_key": "***", "status": 401, "detail": "try again"})
    for _ in range(layers):
        raw, expected = json.dumps(raw), json.dumps(expected)
    assert manager.redact(raw) == expected


@pytest.mark.parametrize("key_type", ["", "RSA ", "EC ", "OPENSSH ", "ENCRYPTED "])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("escaped", [False, True])
def test_private_key_blocks_do_not_leak_or_hide_diagnostics(manager, key_type, complete, escaped):
    raw = "-----BEGIN %sPRIVATE KEY-----\nYWJjZGVm\nYWJjZA==" % key_type
    if complete:
        raw += "\n-----END %sPRIVATE KEY-----" % key_type
    if escaped:
        raw = raw.replace("\n", r"\n")
    assert manager.redact(raw + "\nstatus=401 detail=retry") == "***\nstatus=401 detail=retry"


def test_custom_redactor_runs_after_builtin_redaction(manager):
    manager.register_redactor(r"token=\*\*\*", "[private]")
    assert manager.redact("token=sensitive value") == "[private]"


def test_registered_secret_and_custom_pattern_cover_unlabelled_text(manager):
    manager.register_secret("literal-private-value")
    manager.register_redactor(r"project-private-\d+")
    assert manager.redact("literal-private-value and project-private-123 status=401") == (
        "*** and *** status=401"
    )


@pytest.mark.parametrize("raw", [
    '{"username":"Authorization","value":"public diagnostic"}',
    '{"previous_value":"public diagnostic","name":"Authorization"}',
    '{"ssid":"public wifi","obsession":"keep this diagnostic"}',
])
def test_header_record_field_names_are_exact(manager, raw):
    assert manager.redact(raw) == raw


@pytest.mark.parametrize("name", ["session", "sessionid", "SESSION_ID", "session-id", "session_key", "sessionToken", "sid"])
def test_session_secret_variants(manager, name):
    assert manager.redact('{"%s":"private-value","status":401}' % name) == (
        '{"%s":"***","status":401}' % name
    )
    assert manager.redact("?%s=private-value&status=401" % name) == "?%s=***&status=401" % name


@pytest.mark.parametrize("key", [
    "sk-abcdefghijklmnopqrstuvwxyz1234567890",
    "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
    "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
    "AIzaSy0123456789abcdefghijklmnopqrstuvwxyz",
])
def test_bare_api_keys_preserve_diagnostics(manager, key):
    assert manager.redact("rejected %s; status=401 detail=retry" % key) == "rejected ***; status=401 detail=retry"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("representation", ["json", "repr"])
@pytest.mark.parametrize("layers", [0, 1, 2])
def test_named_headers_allow_metadata_in_any_order(manager, reverse, representation, layers):
    items = [("name", "X-Api-Key"), ("metadata", {"value": "public", "details": ["keep", "{brace}"]}),
             ("value", 'private-value with "quotes"'), ("status", 401)]
    if reverse:
        items.reverse()
    data = {"headers": [dict(items), {"name": "X-Request-Id", "value": "public-request"}]}
    expected = {"headers": [dict((key, "***" if key == "value" else value) for key, value in items),
                            {"name": "X-Request-Id", "value": "public-request"}]}
    stringify = json.dumps if representation == "json" else repr
    raw, result = stringify(data), stringify(expected)
    for _ in range(layers):
        raw, result = json.dumps(raw), json.dumps(result)
    assert manager.redact(raw) == result


def test_named_headers_inside_json_string_values(manager):
    header = {"value": "private-value", "metadata": {"status": "keep"}, "name": "X-Api-Key"}
    raw = json.dumps({"response": json.dumps(header), "status": 401})
    header["value"] = "***"
    assert manager.redact(raw) == json.dumps({"response": json.dumps(header), "status": 401})


def test_named_headers_mask_duplicate_values(manager):
    raw = '{"value":"first-private","metadata":true,"name":"Cookie","value":"second-private","status":401}'
    assert manager.redact(raw) == '{"value":"***","metadata":true,"name":"Cookie","value":"***","status":401}'


@pytest.mark.parametrize("value", ['"private-value"', '["private-value"'])
def test_incomplete_named_header_records_are_redacted(manager, value):
    raw = '{"name":"X-Api-Key","metadata":true,"value":' + value
    assert manager.redact(raw) == '{"name":"X-Api-Key","metadata":true,"value":"***"'


@pytest.mark.parametrize("raw", [
    "basic failure in request path",
    "Basic error while parsing the response",
    "sk-learn raised ValueError",
    "Basic YmFzaWM= is a base64 word, not credentials",
])
def test_ordinary_diagnostics_are_not_bare_credentials(manager, raw):
    assert manager.redact(raw) == raw


@pytest.mark.parametrize("value", ["dXNlcjpwYXNz", "dXNlcjpwYXNzd29yZA==", "YTo=", "YTo"])
def test_bare_basic_auth_requires_encoded_credentials(manager, value):
    assert manager.redact("request used Basic %s; status=401" % value) == "request used Basic ***; status=401"


@pytest.mark.parametrize("newline", ["\n", "\r\n", r"\n", r"\r\n"])
def test_incomplete_encrypted_pem_masks_metadata_and_body(manager, newline):
    raw = newline.join([
        "-----BEGIN RSA PRIVATE KEY-----",
        "Proc-Type: 4,ENCRYPTED",
        "DEK-Info: AES-256-CBC,0123456789ABCDEF0123456789ABCDEF",
        "",
        "YWJjZGVm",
        "YWJjZA==",
    ])
    assert manager.redact(raw + "\nstatus=401 detail=retry") == "***\nstatus=401 detail=retry"


@pytest.mark.parametrize("secret", [
    'prefix"TAIL_SECRET', "prefix'TAIL_SECRET", r'prefix\"TAIL_SECRET',
    "prefix\\'\"TAIL_SECRET", "prefix\\\\TAIL_SECRET",
])
@pytest.mark.parametrize("base", [json.dumps, repr])
@pytest.mark.parametrize("wrappers", [
    (repr, json.dumps), (json.dumps, repr), (repr, json.dumps, repr),
    (json.dumps, repr, json.dumps), (repr, repr, json.dumps, repr),
    (json.dumps, json.dumps, repr, json.dumps),
])
@pytest.mark.parametrize("named_header", [False, True])
def test_mixed_json_repr_layers_use_actual_string_encoding(manager, secret, base, wrappers, named_header):
    if named_header:
        data = {"headers": [{"name": "Cookie", "metadata": {"status": "keep"}, "value": secret}], "status": 401}
        expected = {"headers": [{"name": "Cookie", "metadata": {"status": "keep"}, "value": "***"}], "status": 401}
    else:
        data, expected = {"api_key": secret, "status": 401}, {"api_key": "***", "status": 401}
    raw = base(data)
    for encode in wrappers:
        raw = encode(raw)
    result = manager.redact(raw)
    assert "TAIL_SECRET" not in result
    for encode in reversed(wrappers):
        result = json.loads(result) if encode is json.dumps else ast.literal_eval(result)
    result = json.loads(result) if base is json.dumps else ast.literal_eval(result)
    assert result == expected


@pytest.mark.parametrize("container", [json.dumps, repr])
@pytest.mark.parametrize("encoded", [json.dumps, repr])
def test_mixed_serialized_strings_inside_record_values(manager, container, encoded):
    raw = container({"body": encoded({"api_key": 'prefix"TAIL_SECRET', "status": 401}), "detail": "keep"})
    expected = container({"body": encoded({"api_key": "***", "status": 401}), "detail": "keep"})
    assert manager.redact(raw) == expected


@pytest.mark.parametrize("secret", ['registered"tail', "registered'tail", r"registered\tail", "注册的私密值"])
@pytest.mark.parametrize("wrappers", [
    (json.dumps, repr), (repr, json.dumps), (repr, json.dumps, repr),
    (json.dumps, repr, json.dumps),
])
def test_registered_secrets_inside_mixed_encoded_strings(manager, secret, wrappers):
    manager.register_secret(secret)
    value = {"message": secret, "status": 401, "detail": "keep"}
    raw = json.dumps(value)
    for encode in wrappers:
        raw = encode(raw)
    result = manager.redact(raw)
    for encode in reversed(wrappers):
        result = json.loads(result) if encode is json.dumps else ast.literal_eval(result)
    assert json.loads(result) == {"message": "***", "status": 401, "detail": "keep"}


def test_registered_secret_encoding_preserves_custom_order_and_single_application(manager):
    manager.register_secret('registered"tail')
    manager.register_secret("replace-first")
    manager.register_redactor("replace-first", "custom-result")
    manager.register_redactor("public", "public!")
    raw = repr(json.dumps({"message": 'registered"tail', "other": "replace-first", "detail": "public"}))
    result = json.loads(ast.literal_eval(manager.redact(raw)))
    assert result == {"message": "***", "other": "custom-result", "detail": "public!"}


def test_registered_secret_in_encoded_scalar_string(manager):
    manager.register_secret('registered"tail')
    raw = repr(json.dumps('registered"tail'))
    assert json.loads(ast.literal_eval(manager.redact(raw))) == "***"


@pytest.mark.parametrize("encode", [json.dumps, repr])
def test_named_headers_with_unstructured_error_prefix(manager, encode):
    raw = "failure: " + encode({"headers": [{"name": "Cookie", "value": "TAIL_SECRET"}]})
    expected = "failure: " + encode({"headers": [{"name": "Cookie", "value": "***"}]})
    assert manager.redact(raw) == expected


@pytest.mark.parametrize("named_header", [False, True])
def test_truncated_outer_repr_does_not_leak_or_claim_closing_quote(manager, named_header):
    data = {"api_key": 'prefix"TAIL_SECRET', "status": 401}
    expected = {"api_key": "***", "status": 401}
    if named_header:
        data = {"headers": [{"name": "Cookie", "value": 'prefix"TAIL_SECRET'}], "status": 401}
        expected = {"headers": [{"name": "Cookie", "value": "***"}], "status": 401}
    raw = "failure: " + repr(json.dumps(data))[:-1]
    assert manager.redact(raw) == "failure: " + repr(json.dumps(expected))[:-1]


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


@pytest.mark.parametrize("prefix", ["", "prefix_", "context.part-" * 20])
@pytest.mark.parametrize("key", ["token", "PassWord", "api key", "private-key", "ſecret"])
@pytest.mark.parametrize("delimiter", [" = ", ": "])
def test_redaction_preserves_prefix_and_key(manager, prefix, key, delimiter):
    assert manager.redact(prefix + key + delimiter + "sensitive value\nnext line") == (
        prefix + key + delimiter + "***\nnext line"
    )


def test_redaction_uses_first_sensitive_value_delimiter(manager):
    assert manager.redact("token=first password=second") == "token=***"
    assert manager.redact("token-password=second") == "token-password=***"


def test_no_redos_on_long_input(manager):
    import time as _time
    start = _time.monotonic()
    assert manager.redact("a" * 100000) == "a" * 100000
    elapsed = _time.monotonic() - start
    # Match sensitive keys directly without retrying arbitrary prefixes.
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
