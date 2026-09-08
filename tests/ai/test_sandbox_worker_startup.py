#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bubblewrap worker startup diagnostic regressions."""

from linktools.ai.workspace import sandbox_worker
from linktools.ai.workspace._sandbox_protocol import WORKER_EXIT_SESSION_FAILED


def test_worker_reports_pre_handshake_failure(capsys) -> None:
    result = sandbox_worker.main(
        [
            "--resources-json",
            "[]",
            "--worker-build",
            "unsupported-build",
            "--lock-root",
            "/__linktools_locks",
        ]
    )

    assert result == WORKER_EXIT_SESSION_FAILED
    captured = capsys.readouterr()
    assert "sandbox worker failed: RuntimeError: worker build is unsupported" in (
        captured.err
    )
