#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage selection keeps unclassified tests and isolates manual probes."""

import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import manage

from scripts.check.matrix import package_checks
from scripts.check.tiers import select_ci_tier

pytest_plugins = ("pytester",)


@pytest.mark.parametrize("draft,expected", ((True, "daily"), (False, "merge")))
def test_pull_request_coverage_tracks_current_draft_state(draft: bool, expected: str) -> None:
    assert select_ci_tier("pull_request", draft, "") == expected


@pytest.mark.parametrize("event", ("push", "workflow_call", "release", "schedule", "workflow_dispatch"))
def test_automatic_and_reused_workflows_exclude_manual_probes(event: str) -> None:
    # Reusable workflows inherit the caller event but have no test-tier input.
    assert select_ci_tier(event, False, "") == "merge"


@pytest.mark.parametrize("tier", ("daily", "merge", "all"))
def test_explicit_manual_dispatch_selects_coverage(tier: str) -> None:
    assert select_ci_tier("workflow_dispatch", False, tier) == tier


@pytest.mark.parametrize("event,tier", (("push", "all"), ("pull_request", "all"), ("workflow_call", "all"), ("workflow_dispatch", "typo")))
def test_invalid_manual_selection_fails_closed(event: str, tier: str) -> None:
    with pytest.raises(ValueError):
        select_ci_tier(event, False, tier)


@pytest.mark.parametrize("tier,passed,deselected", ((None, 3, 2), ("daily", 2, 3), ("merge", 3, 2), ("all", 5, 0)))
def test_collection_keeps_unknown_tests_and_never_leaks_manual_cases(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
    tier: "str | None", passed: int, deselected: int,
) -> None:
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini(
        (root / "pytest.ini").read_text(encoding="utf-8")
        + "    new_category: unrelated test category\n"
    )
    pytester.makepyfile('''
import pytest

def test_unmarked(): pass
@pytest.mark.new_category
def test_new_category(): pass
@pytest.mark.merge
def test_backend_matrix(): pass
@pytest.mark.manual
def test_scale(): pass
@pytest.mark.merge
@pytest.mark.manual
def test_manual_takes_precedence(): pass
''')
    args = ["-q", "-p", "no:asyncio"]
    if tier is not None:
        args.extend(("--test-tier", tier))
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=passed, deselected=deselected, warnings=0)


def test_ci_matrix_retains_new_packages_and_partitions_ai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path(manage.PROJECT_PATH)
    for package in ("linktools", "linktools-ai", "linktools-new"):
        folder = tmp_path / package
        folder.mkdir()
        (folder / "tests").mkdir()
        data = {"checks": {"pytest": {"paths": ["tests"]}}}
        if package == "linktools-ai":
            data["checks"]["pytest"]["groups"] = yaml.safe_load(
                (original / package / "linktools.yml").read_text()
            )["checks"]["pytest"]["groups"]
        (folder / "linktools.yml").write_text(yaml.safe_dump(data, sort_keys=False))
    monkeypatch.setattr(manage, "PROJECT_PATH", str(tmp_path))
    checks = package_checks(("linktools", "linktools-ai", "linktools-new"))
    assert [(check["package"], check["group"]) for check in checks] == [
        ("linktools", "all"), ("linktools-ai", "evaluation"),
        ("linktools-ai", "runtime"), ("linktools-new", "all"),
    ]
    assert len({check["name"] for check in checks}) == len(checks)
    assert checks[0]["name"] == "linktools checks"
    assert checks[-1]["name"] == "linktools-new checks"
    assert checks[0]["install"] == ""
    assert all(check["install"] == check["package"] for check in checks[1:])


@pytest.mark.parametrize("tier", ("daily", "merge", "all"))
@pytest.mark.parametrize("group", ("all", "evaluation", "runtime"))
def test_ai_groups_partition_file_families_without_changing_tiers(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tier: str, group: str,
) -> None:
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini((root / "pytest.ini").read_text(encoding="utf-8"))
    check = manage.load_project_checks("linktools-ai", str(root / "linktools-ai"))["pytest"]
    check["paths"] = [str(pytester.path / "tests/ai"), str(pytester.path / "linktools-ai/tests")]
    monkeypatch.setenv("LINKTOOLS_PYTEST", json.dumps(check))
    families = {
        "tests/ai/test_evaluation_new.py": "evaluation",
        "tests/ai/test_new_capture.py": "runtime",
        "tests/ai/test_captured_new.py": "evaluation",
        "tests/ai/test_graph_capture_new.py": "evaluation",
        "tests/ai/test_new_feature.py": "runtime",
        "linktools-ai/tests/test_evaluation_local.py": "evaluation",
        "linktools-ai/tests/test_local_capture.py": "runtime",
        "linktools-ai/tests/test_local_feature.py": "runtime",
        "tests/core/test_evaluation_other.py": "all",
    }
    for name in families:
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('''
import pytest
def test_daily(): pass
@pytest.mark.merge
def test_merge(): pass
@pytest.mark.manual
def test_manual(): pass
''', encoding="utf-8")
    result = pytester.runpytest(
        "-q", "-p", "no:asyncio", "--collect-only", "--test-tier", tier, "--test-group", group,
    )
    assert result.ret == 0
    selected = {line for line in result.outlines if line.startswith(("tests/", "linktools-ai/tests/")) and "::test_" in line}
    cases = ("daily",) if tier == "daily" else (("daily", "merge") if tier == "merge" else ("daily", "merge", "manual"))
    assert selected == {
        "%s::test_%s" % (name, case)
        for name, family in families.items()
        if group == "all" or family in (group, "all")
        for case in cases
    }
    assert "warning" not in result.stdout.str().lower()


def test_empty_ai_group_fails_instead_of_reporting_coverage(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    check = manage.load_project_checks("linktools-ai", str(root / "linktools-ai"))["pytest"]
    check["paths"] = [str(pytester.path / "tests/ai")]
    monkeypatch.setenv("LINKTOOLS_PYTEST", json.dumps(check))
    path = pytester.path / "tests" / "ai" / "test_runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text("def test_runtime(): pass\n", encoding="utf-8")
    result = pytester.runpytest("-q", "-p", "no:asyncio", "--test-group", "evaluation")
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_generated_ci_plan_keeps_execution_and_compatibility_versions_in_sync() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(subprocess.check_output(
        [sys.executable, "-m", "scripts.check.matrix", json.dumps(["linktools", "linktools-ai"])],
        cwd=root, text=True,
    ))
    assert plan["python-versions"] == ["3.10", "3.x"]
    for check in plan["checks"]:
        options = shlex.split(check["pytest-args"])
        assert options[:2] == ["-n", "4"]
        assert "--dist=loadfile" in options
        assert "--test-group=" + check["group"] in options
        assert "-rs" in options
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    jobs = workflow["jobs"]
    assert jobs["python"]["strategy"]["matrix"]["python-version"] == jobs["ai_coverage"]["strategy"]["matrix"]["python-version"]
    assert jobs["python"]["strategy"]["fail-fast"] is False
    assert jobs["ai_coverage"]["name"] == "Python ${{ matrix.python-version }} linktools-ai checks"
    assert jobs["coverage"]["name"] == "Python test coverage"


@pytest.mark.parametrize("job", ("ai_coverage", "coverage"))
def test_ci_aggregates_reject_every_incomplete_dependency(tmp_path: Path, job: str) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    definition = workflow["jobs"][job]
    assert definition["if"] == "${{ always() }}"
    step = definition["steps"][0]
    results = [key for key in step["env"] if key.endswith("_RESULT")]
    assert len(results) == len(definition["needs"])
    for states in itertools.product(("success", "failure", "cancelled", "skipped"), repeat=len(results)):
        environment = dict(os.environ, TEST_TIER="merge", GITHUB_STEP_SUMMARY=str(tmp_path / "summary"))
        environment.update(zip(results, states))
        outcome = subprocess.run(["bash", "-e", "-c", step["run"]], env=environment, capture_output=True, check=False)
        assert (outcome.returncode == 0) == all(state == "success" for state in states)
