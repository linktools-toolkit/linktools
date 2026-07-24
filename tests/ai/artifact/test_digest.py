#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactDigest value-object validation: a digest must be exactly 64 lowercase
hex characters before it can become a coordination key, a lock-file path, or a
blob address. Every other shape is rejected at the boundary; the rejected value
is never echoed in the error message."""

import hashlib

import pytest

from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.errors import InvalidArtifactDigestError

_VALID = "0123456789abcdef" * 4  # exactly 64 lowercase hex chars


def test_valid_lowercase_hex_succeeds():
    digest = ArtifactDigest(_VALID)
    assert digest.value == _VALID
    assert str(digest) == _VALID


def test_parse_roundtrips_value():
    digest = ArtifactDigest.parse(_VALID)
    assert digest.value == _VALID


def test_from_bytes_matches_hashlib():
    content = b"some artifact bytes"
    digest = ArtifactDigest.from_bytes(content)
    assert digest.value == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "bad",
    [
        "0" * 63,          # too short
        "0" * 65,          # too long
        "A" * 64,          # uppercase hex
        "g" * 64,          # non-hex letter
        "../x",            # traversal
        "/tmp/x",          # absolute path
        "\\" * 64,         # backslash separator
        "." * 64,          # dot component
        " " * 64,          # whitespace
        "\n" * 64,         # newline
        "0" * 63 + "\n",   # trailing newline
        "0" * 63 + " ",    # trailing space
        "é" * 64,          # unicode (non-ascii)
        "0" * 32 + " " + "0" * 31,  # embedded space
    ],
    ids=[
        "too-short", "too-long", "uppercase", "non-hex", "traversal", "abs-path",
        "backslash", "dots", "spaces", "newlines", "trailing-newline",
        "trailing-space", "unicode", "embedded-space",
    ],
)
def test_invalid_shapes_are_rejected(bad):
    with pytest.raises(InvalidArtifactDigestError):
        ArtifactDigest(bad)


@pytest.mark.parametrize(
    "bad", ["../x", "/tmp/x", "ABCD" + "0" * 60]
)
def test_parse_rejects_bad_input(bad):
    with pytest.raises(InvalidArtifactDigestError):
        ArtifactDigest.parse(bad)


def test_error_message_does_not_echo_value():
    malicious = "../etc/passwd"
    try:
        ArtifactDigest(malicious)
    except InvalidArtifactDigestError as exc:
        assert malicious not in str(exc)
    else:
        pytest.fail("expected InvalidArtifactDigestError")


def test_value_object_is_hashable_and_frozen():
    d1 = ArtifactDigest(_VALID)
    d2 = ArtifactDigest(_VALID)
    assert d1 == d2
    assert hash(d1) == hash(d2)
    with pytest.raises(Exception):
        d1.value = "x"  # type: ignore[misc]  # frozen dataclass
