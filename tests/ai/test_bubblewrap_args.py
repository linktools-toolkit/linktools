#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bubblewrap launcher argument regressions."""

from pathlib import Path

from linktools.ai.workspace._bubblewrap import _build_bwrap_args


def test_network_setup_keeps_bubblewrap_internal_capabilities(tmp_path: Path) -> None:
    args = _build_bwrap_args(
        root=tmp_path / "workspace",
        runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"),
        lock_root=tmp_path / "locks",
        resources=(),
        hidden_paths=(),
        worker_resources=[],
    )

    assert "--unshare-net" in args
    assert "--cap-drop" not in args
