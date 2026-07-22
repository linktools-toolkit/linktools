#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary freeze for the task / evaluation / artifact extension.

of the infrastructure plan establishes architecture invariants BEFORE
any task/evaluation/artifact code lands, so the extension cannot silently
change the existing runtime's public surface or storage shape:

* the ``linktools.ai`` root still exports exactly ``Runtime`` -- nothing else;
* importing the root (in a fresh interpreter) does not pull in the new domains;
* the existing ``Storage`` facade field set is snapshotted, so adding the
  optional ``jobs`` / ``evaluations`` fields later is a visible, intentional
  change rather than an accident;
* the rejected ``linktools.ai.durable`` namespace stays absent.

When a legitimately changes one of these (e.g. phase 3 adds
``Storage.jobs``), update the snapshot here in the same change.
"""

import dataclasses
import subprocess
import sys


def test_root_api_exports_exactly_runtime() -> None:
    import linktools.ai

    assert list(linktools.ai.__all__) == ["Runtime"]
    assert hasattr(linktools.ai, "Runtime")


def test_importing_root_in_fresh_process_does_not_load_new_domains() -> None:
    # a fresh interpreter importing the root must not load
    # the new domains (they must not exist / not auto-load). Run in a
    # subprocess so other tests' imports cannot pollute the check.
    code = (
        "import sys\n"
        "import linktools.ai\n"
        "assert list(linktools.ai.__all__) == ['Runtime']\n"
        "for mod in ('linktools.ai.jobs','linktools.ai.evaluation',"
        "'linktools.ai.artifact'):\n"
        "    assert mod not in sys.modules, mod\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_rejected_durable_namespace_stays_absent() -> None:
    import importlib

    try:
        importlib.import_module("linktools.ai.durable")
    except ModuleNotFoundError:
        return
    raise AssertionError("linktools.ai.durable must not exist (rejected approach)")


def test_storage_facade_field_set_snapshot() -> None:
    from linktools.ai.storage.facade import Storage

    fields = {f.name for f in dataclasses.fields(Storage)}
    # The facade. added optional `jobs` (renamed from `tasks`); phase
    # 8 added optional `evaluations`. Both default to None for backward
    # compatibility. StorageCapabilities was converged to StorageFeatures
    # (the capability surface is now scoped enums + first-class
    # streaming/fencing flags). `coordination` is the LeaseCoordinator field
    # (process-local reference; downstream injects a distributed one and
    # declares it on StorageFeatures).
    assert fields == {
        "assets",
        "sessions",
        "runs",
        "events",
        "checkpoints",
        "swarms",
        "memories",
        "approvals",
        "idempotency",
        "features",
        "coordination",
        "_transaction_manager",
        "run_definitions",
        "jobs",
        "evaluations",
        "artifacts",
    }


def test_existing_storage_backends_remain_importable() -> None:
    # The extension reuses (does not replace) the existing storage backends.
    from linktools.ai.storage import filesystem as file_storage
    from linktools.ai.storage import sqlalchemy as sa_storage
    from linktools.ai.asset.store import AssetStore

    assert file_storage is not None
    assert sa_storage is not None
    assert AssetStore is not None


def test_importing_jobs_does_not_load_sqlalchemy_or_langgraph() -> None:
    # (second block): importing the jobs domain keeps the
    # optional heavy deps out of sys.modules.
    code = (
        "import sys\n"
        "import linktools.ai.jobs.models\n"
        "for mod in ('sqlalchemy', 'langgraph'):\n"
        "    assert mod not in sys.modules, mod\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_no_runtime_internals_or_pydantic_ai_in_new_domains() -> None:
    # jobs/ and evaluation/ must not reach into the runtime
    # internals; jobs/ must not import pydantic_ai.
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "linktools-ai" / "src" / "linktools" / "ai"
    for sub in ("jobs", "evaluation"):
        for p in (root / sub).rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            assert "linktools.ai.runtime.builder" not in text, f"_runtime leak in {p}"
    for p in (root / "jobs").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "pydantic_ai" not in text, f"pydantic_ai leak in {p}"
