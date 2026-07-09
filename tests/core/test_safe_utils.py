# -*- coding: utf-8 -*-
"""Tests for safe_remove/safe_rmtree (§17.5) and verify_file (§17.3)."""
import pytest

from linktools import utils
from linktools.errors import LinktoolsError


# §17.5 ----------------------------------------------------------------------

def test_safe_remove_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert utils.safe_remove(f, root=tmp_path) is True
    assert not f.exists()


def test_safe_remove_dir(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "f.txt").write_text("x")
    assert utils.safe_rmtree(d, root=tmp_path) is True
    assert not d.exists()


def test_safe_remove_absent_returns_false(tmp_path):
    assert utils.safe_remove(tmp_path / "nope", root=tmp_path) is False


def test_safe_remove_outside_root_rejected(tmp_path):
    outside = tmp_path.parent / "sibling-target"
    outside.mkdir(exist_ok=True)
    try:
        with pytest.raises(LinktoolsError):
            utils.safe_remove(outside, root=tmp_path)
        assert outside.exists()  # untouched
    finally:
        outside.rmdir()  # cleanup


def test_safe_remove_default_root_is_parent(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("x")
    # root defaults to parent -> allowed.
    assert utils.safe_remove(f) is True


# §17.3 ----------------------------------------------------------------------

def test_verify_file_matches(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"hello")
    digest = utils.get_file_hash(f, algorithm="sha256")
    assert utils.verify_file(f, digest, algorithm="sha256") is True


def test_verify_file_mismatch(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"hello")
    assert utils.verify_file(f, "0" * 64, algorithm="sha256") is False
