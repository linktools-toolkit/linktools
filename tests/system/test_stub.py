# -*- coding: utf-8 -*-
"""Tests for linktools.system.stub.CommandStub.

Verifies generated wrapper content (POSIX + Windows .bat), atomic writes
(no half-written file / no leftover temp under concurrent writers), the
``exists``/``remove`` lifecycle, and -- on POSIX -- a real execution
round-trip proving fixed args + caller args + spaces + quotes + exit code
all survive intact.
"""
import os
import stat
import subprocess
import sys
import threading

import pytest

from linktools.system.stub import CommandStub


# -- rendered content ----------------------------------------------------

def test_posix_render_uses_bin_sh_and_exec(tmp_path):
    stub = CommandStub(tmp_path, "adb", system="linux")
    stub.write(["/p/python", "-m", "lt", "tool", "adb"])
    text = stub.path.read_text()
    assert text == "#!/bin/sh\nexec '/p/python' '-m' 'lt' 'tool' 'adb' \"$@\"\n"


def test_posix_name_has_no_suffix(tmp_path):
    assert CommandStub(tmp_path, "adb", system="linux").name == "adb"


def test_windows_render_is_bat_with_errorlevel(tmp_path):
    stub = CommandStub(tmp_path, "adb", system="windows")
    stub.write([r"C:\py", "-m", "lt"])
    text = stub.path.read_text()
    assert stub.name == "adb.bat"
    assert text.startswith("@echo off\n")
    assert '"C:\\py" "-m" "lt" %*' in text
    assert text.rstrip().endswith("exit /b %ERRORLEVEL%")


def test_write_rejects_string_argv(tmp_path):
    with pytest.raises(TypeError):
        CommandStub(tmp_path, "x").write("/p/py -m lt")


def test_write_rejects_empty_argv(tmp_path):
    with pytest.raises(ValueError):
        CommandStub(tmp_path, "x").write([])


# -- lifecycle -----------------------------------------------------------

def test_exists_and_remove(tmp_path):
    stub = CommandStub(tmp_path, "adb")
    assert stub.exists is False
    stub.write(["/p/py"])
    assert stub.exists is True
    stub.remove()
    assert stub.exists is False
    stub.remove()  # idempotent on missing file


def test_posix_write_sets_executable_bit(tmp_path):
    stub = CommandStub(tmp_path, "adb")
    stub.write(["/p/py"])
    mode = stat.S_IMODE(os.stat(stub.path).st_mode)
    assert mode & 0o755 == 0o755


def test_write_creates_parent_directory(tmp_path):
    stub = CommandStub(tmp_path / "deep" / "nest", "adb")
    stub.write(["/p/py"])
    assert stub.exists


# -- real POSIX execution round-trip -------------------------------------

def _printer_code():
    # prints repr(sys.argv[1:]) so the test can compare exact forwarded argv
    return "import sys,os; os.write(1, repr(sys.argv[1:]).encode())"


def test_posix_stub_forwards_fixed_and_caller_args(tmp_path):
    stub = CommandStub(tmp_path, "printer")
    stub.write([sys.executable, "-c", _printer_code(), "FIXED"])
    proc = subprocess.run(
        [str(stub.path), "hello world", "a'b", "$HOME"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # fixed arg + three caller args survive verbatim; $HOME stays literal
    assert proc.stdout == repr(["FIXED", "hello world", "a'b", "$HOME"])


def test_posix_stub_propagates_exit_code(tmp_path):
    stub = CommandStub(tmp_path, "rc")
    code = "import sys; sys.exit(42)"
    stub.write([sys.executable, "-c", code])
    proc = subprocess.run([str(stub.path)], capture_output=True)
    assert proc.returncode == 42


# -- concurrency: atomic, no half-file, no leftover temp -----------------

def test_concurrent_writers_leave_complete_file_and_no_temp(tmp_path):
    stub = CommandStub(tmp_path, "adb")
    argv = [sys.executable, "-c", _printer_code(), "x"]

    def writer():
        for _ in range(20):
            stub.write(argv)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # final file is complete, executable, and correct
    assert stub.exists
    assert os.access(stub.path, os.X_OK)
    proc = subprocess.run([str(stub.path), "y"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == repr(["x", "y"])
    # no leftover .tmp files in the directory
    assert not list(tmp_path.glob("*.tmp"))
