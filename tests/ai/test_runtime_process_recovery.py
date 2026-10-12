#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh processes recover orphaned execution and prepared terminal handoff."""

from pathlib import Path

import pytest

from ._runtime_process_test_helpers import _RuntimeProcess


@pytest.mark.parametrize("phase", (
    "active", pytest.param("admitted", marks=pytest.mark.merge), "pending",
))
def test_sql_orphan_waits_for_explicit_recovery(tmp_path: Path, phase: str) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase=phase)
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase=phase, execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.initial["model_calls"] == 0
        remote.request("recover")
        resumed = remote.request("await_model")
        assert resumed["generation"] > owner.initial["generation"]
        assert resumed["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_explicit_recovery_finishes_orphaned_cancel_without_restarting_model(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", execution_id=owner.initial["execution_id"],
        )
        cancelled = remote.request("cancel")
        assert cancelled["status"] == "CANCELLING"
        assert cancelled["cancelled"] is False
        owner.crash()
        recovered = remote.request("recover")
        assert recovered["status"] == "CANCELLED"
        assert remote.request("result") == {
            "status": "CANCELLED", "output": None, "model_calls": 0,
        }
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_sql_startup_completes_a_prepared_terminal_handoff(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="handoff")
    remote = None
    try:
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="handoff", execution_id=owner.initial["execution_id"],
        )
        assert owner.process.is_alive()
        assert remote.initial["status"] == "SUCCEEDED"
        assert remote.initial["generation"] == owner.initial["generation"]
        assert remote.request("result") == {
            "status": "SUCCEEDED", "output": {"text": "owner answer"}, "model_calls": 0,
        }
        assert owner.request("finish")["status"] == "SUCCEEDED"
        owner.close(check=True)
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()


def test_filesystem_execution_route_preserves_automatic_crash_recovery(tmp_path: Path) -> None:
    owner = _RuntimeProcess(tmp_path / "runtime.db", phase="filesystem")
    remote = None
    try:
        owner.crash()
        remote = _RuntimeProcess(
            tmp_path / "runtime.db", phase="filesystem", execution_id=owner.initial["execution_id"],
        )
        assert remote.initial["generation"] > owner.initial["generation"]
        assert remote.request("await_model")["model_calls"] == 1
        assert remote.request("finish")["status"] == "SUCCEEDED"
        remote.close(check=True)
    finally:
        owner.close()
        if remote is not None:
            remote.close()
