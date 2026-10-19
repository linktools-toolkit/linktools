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
        ("linktools", "all"), ("linktools-ai", "evaluation-recovery"),
        ("linktools-ai", "runtime"), ("linktools-new", "all"),
    ]
    assert len({check["name"] for check in checks}) == len(checks)
    assert checks[0]["name"] == "linktools checks"
    assert checks[-1]["name"] == "linktools-new checks"
    assert checks[0]["install"] == ""
    assert all(check["install"] == check["package"] for check in checks[1:])


@pytest.mark.parametrize("tier", ("daily", "merge", "all"))
@pytest.mark.parametrize("group", ("all", "evaluation-recovery", "runtime"))
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
        "tests/ai/test_evaluation_new.py": "evaluation-recovery",
        "tests/ai/test_new_capture.py": "runtime",
        "tests/ai/test_captured_new.py": "evaluation-recovery",
        "tests/ai/test_graph_capture_new.py": "evaluation-recovery",
        "tests/ai/test_runtime_recovery_ownership.py": "evaluation-recovery",
        "tests/ai/test_runtime_process_recovery.py": "evaluation-recovery",
        "tests/ai/test_keyed_execution_recovery.py": "evaluation-recovery",
        "tests/ai/test_runtime_other.py": "runtime",
        "tests/ai/test_new_feature.py": "runtime",
        "linktools-ai/tests/test_evaluation_local.py": "evaluation-recovery",
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
    result = pytester.runpytest("-q", "-p", "no:asyncio", "--test-group", "evaluation-recovery")
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_generated_ci_plan_supplies_execution_versions_and_options() -> None:
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
        assert "--capture=fd" in options
        assert "--test-group=" + check["group"] in options
        assert "-vv" not in options
        assert ("--durations=50" in options) == (check["group"] != "all")
        if check["group"] == "runtime":
            assert "-ra" in options
            assert options[-5:] == ["--tb=long", "-o", "verbosity_assertions=2", "-o", "faulthandler_timeout=90"]
        else:
            assert "-rs" in options
            assert not any(option.startswith("faulthandler_timeout=") for option in options)
        assert "--showlocals" not in options
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    jobs = workflow["jobs"]
    assert jobs["python"]["strategy"]["matrix"]["python-version"] == "${{ fromJSON(needs.discover.outputs.plan).python-versions }}"
    assert jobs["python"]["strategy"]["fail-fast"] is False
    assert jobs["coverage"]["name"] == "Python checks passed"


def test_runtime_ci_output_keeps_failures_without_per_case_progress(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini((root / "pytest.ini").read_text(encoding="utf-8"))
    check = manage.load_project_checks("linktools-ai", str(root / "linktools-ai"))["pytest"]
    check["paths"] = [str(pytester.path)]
    monkeypatch.setenv("LINKTOOLS_PYTEST", json.dumps(check))
    row = next(row for row in package_checks(("linktools-ai",)) if row["group"] == "runtime")
    monkeypatch.setenv("PYTEST_ADDOPTS", row["pytest-args"])
    pytester.makepyfile(test_runtime_output='''
import warnings
import pytest

@pytest.mark.parametrize("value", range(20))
def test_pass(value):
    print("passing-captured-output")
    if value == 0:
        warnings.warn("visible-runtime-warning", UserWarning)

def assert_contents():
    actual = list(range(30))
    actual[15] = 999
    assert actual == list(range(30))

def test_fail():
    print("failing-captured-output")
    assert_contents()

def test_skip():
    pytest.skip("visible-runtime-skip")
''')
    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider", "--test-tier", "daily")
    result.assert_outcomes(passed=20, failed=1, skipped=1, warnings=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    output = result.stdout.str()
    progress, failures = output.split(" FAILURES ", 1)
    assert "test_pass" not in progress
    assert "[gw" not in progress
    assert "passing-captured-output" not in output
    assert "failing-captured-output" in failures
    assert "assert_contents()" in failures
    assert "Full diff:" in failures
    assert "Full output truncated" not in failures
    assert "999" in failures
    assert "visible-runtime-warning" in failures
    assert "visible-runtime-skip" in failures
    assert "FAILED test_runtime_output.py::test_fail" in failures
    assert "slowest 50 durations" in failures


def test_ci_pip_cache_distinguishes_installation_sets(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    steps = workflow["jobs"]["python"]["steps"]
    setup_index = next(index for index, step in enumerate(steps) if step.get("with", {}).get("cache") == "pip")
    patterns = steps[setup_index]["with"]["cache-dependency-path"].splitlines()
    assert {"requirements.txt", "linktools*/linktools.yml"}.issubset(patterns)
    (tmp_path / ".github").mkdir()
    original_inputs = set()
    for pattern in patterns:
        for source in root.glob(pattern):
            if source.is_file():
                target = tmp_path / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                original_inputs.add(target)

    def cache_inputs() -> tuple:
        paths = {path for pattern in patterns for path in tmp_path.glob(pattern) if path.is_file()}
        return tuple((str(path.relative_to(tmp_path)), path.read_bytes()) for path in sorted(paths))

    preparations = [step for step in steps[:setup_index]
                    if step.get("env", {}).get("INSTALL_PACKAGES") == "${{ matrix.check.install }}"]
    original_snapshot = cache_inputs()
    snapshots = []
    for packages in ("first second", "  second\tfirst first  ", "", "first"):
        environment = dict(os.environ, INSTALL_PACKAGES=packages)
        for step in preparations:
            subprocess.run(["bash", "-eo", "pipefail", "-c", step["run"]],
                           cwd=tmp_path, env=environment, check=True)
        snapshots.append(cache_inputs())
    assert snapshots[0] == snapshots[1]
    assert len({snapshots[0], snapshots[2], snapshots[3]}) == 3

    install = next(step for step in steps[setup_index + 1:]
                   if step.get("env", {}).get("INSTALL_PACKAGES") == "${{ matrix.check.install }}")
    generated_inputs = {tmp_path / path for path, _ in cache_inputs()} - original_inputs
    stub = tmp_path / "python"
    stub.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    environment = dict(os.environ, INSTALL_PACKAGES="first",
                       PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    subprocess.run(["bash", "-eo", "pipefail", "-c", install["run"]],
                   cwd=tmp_path, env=environment, check=True)
    assert not any(path.exists() for path in generated_inputs)
    assert cache_inputs() == original_snapshot


def test_ci_coverage_rejects_every_incomplete_dependency(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    definition = workflow["jobs"]["coverage"]
    assert definition["needs"] == ["discover", "python36", "python"]
    assert definition["if"] == "${{ always() }}"
    step = definition["steps"][0]
    results = [key for key in step["env"] if key.endswith("_RESULT")]
    assert {step["env"][key] for key in results} == {
        "${{ needs.%s.result }}" % job for job in definition["needs"]
    }
    for states in itertools.product(("success", "failure", "cancelled", "skipped", ""), repeat=len(results)):
        environment = dict(os.environ, TEST_TIER="merge", GITHUB_STEP_SUMMARY=str(tmp_path / "summary"))
        environment.update(zip(results, states))
        outcome = subprocess.run(["bash", "-e", "-c", step["run"]], env=environment, capture_output=True, check=False)
        assert (outcome.returncode == 0) == all(state == "success" for state in states)


@pytest.mark.parametrize("code,signal", (
    (0, ""), (1, ""), (2, ""), (124, ""), (130, ""), (143, ""), (130, "INT"), (143, "TERM"),
))
@pytest.mark.parametrize("failed_index", (0, 1))
def test_ci_package_loop_preserves_failures_and_stops_on_signals(
    tmp_path: Path, code: int, signal: str, failed_index: int,
) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    step = next(step for step in workflow["jobs"]["python"]["steps"] if step.get("id") == "checks")
    stub = tmp_path / "python"
    stub.write_text(
        "#!/bin/bash\n"
        "printf '%s|%s\\n' \"$*\" \"$PYTEST_ADDOPTS\" >> \"$CALLS\"\n"
        "if [ \"$3\" = \"$FAILED_PACKAGE\" ]; then\n"
        "  if [ -n \"$SIGNAL\" ]; then kill -s \"$SIGNAL\" \"$PPID\"; fi\n"
        "  exit \"$EXIT_CODE\"\n"
        "fi\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    packages = ("linktools", "linktools-common", "linktools-mobile")
    options = "-n 4 --dist=loadfile --capture=fd -rs --test-group=all"
    calls = tmp_path / "calls"
    summary = tmp_path / "summary"
    environment = dict(
        os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        PACKAGES=" ".join(packages), PYTEST_ADDOPTS=options, PYTHON="3.10", GROUP="all",
        TEST_TIER="merge", GITHUB_STEP_SUMMARY=str(summary), CALLS=str(calls),
        FAILED_PACKAGE=packages[failed_index], EXIT_CODE=str(code), SIGNAL=signal,
    )
    outcome = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert outcome.returncode == (code if code >= 128 else int(code != 0))
    attempted = packages[:failed_index + 1] if code >= 128 else packages
    assert calls.read_text().splitlines() == [
        "manage.py check %s --skip-compatibility --test-tier merge|%s" % (package, options)
        for package in attempted
    ]
    summarized = attempted[:-1] if signal else attempted
    assert (summary.read_text().splitlines() if summary.exists() else []) == [
        "%s / Python 3.10 / all / merge: %s" % (
            package, "failure" if code and package == packages[failed_index] else "success",
        ) for package in summarized
    ]
    assert outcome.stdout.count("::group::") == len(attempted)
    assert outcome.stdout.count("::endgroup::") == len(summarized)


def test_ci_package_loop_rejects_an_empty_plan(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    step = next(step for step in workflow["jobs"]["python"]["steps"] if step.get("id") == "checks")
    environment = dict(os.environ, PACKAGES="", GITHUB_STEP_SUMMARY=str(tmp_path / "summary"))
    outcome = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]], env=environment, capture_output=True, check=False,
    )
    assert outcome.returncode != 0


@pytest.mark.parametrize("child,expected", (
    ("raise SystemExit(1)", 1),
    ("raise SystemExit(2)", 1),
    ("raise SystemExit(130)", 130),
    ("raise SystemExit(143)", 143),
    ("import os, signal; os.kill(os.getpid(), signal.SIGINT)", 130),
    ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", 143),
))
def test_manage_preserves_interrupted_gate_exit_status(child: str, expected: int) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        "import manage, sys\n"
        "manage._run_ruff = lambda project, check, environment: "
        "manage._run_check([sys.executable, '-c', %r], environment)\n"
        "sys.argv = ['manage.py', 'check', 'linktools-common', '--skip-compatibility']\n"
        "manage.main()\n"
    ) % child
    outcome = subprocess.run(
        [sys.executable, "-c", source], cwd=root, capture_output=True, text=True, check=False,
    )
    assert outcome.returncode == expected, outcome.stderr
    if expected == 1:
        assert "CalledProcessError" in outcome.stderr


def test_ci_summary_distinguishes_job_failure_from_package_results(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/python-check.yml").read_text())
    step = next(step for step in workflow["jobs"]["python"]["steps"] if step.get("name") == "Summarize check")
    summary = tmp_path / "summary"
    summary.write_text("linktools / Python 3.10 / all / merge: success\n", encoding="utf-8")
    environment = dict(
        os.environ, PACKAGE="linktools", PYTHON="3.10", GROUP="all", TEST_TIER="merge",
        RESULT="failure", GITHUB_STEP_SUMMARY=str(summary),
    )
    subprocess.run(["bash", "-e", "-c", step["run"]], env=environment, check=True)
    assert summary.read_text().splitlines() == [
        "linktools / Python 3.10 / all / merge: success",
        "Job linktools / Python 3.10 / all / merge: failure",
    ]
