#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage selection keeps unclassified tests and isolates manual probes."""

from pathlib import Path

import pytest

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
    pytester: pytest.Pytester, tier: "str | None", passed: int, deselected: int,
) -> None:
    root = Path(__file__).resolve().parents[2]
    pytester.makeconftest((root / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini((root / "pytest.ini").read_text(encoding="utf-8"))
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
    args = ["-q"]
    if tier is not None:
        args.extend(("--test-tier", tier))
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=passed, deselected=deselected)
