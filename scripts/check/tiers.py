#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select the CI test tier without changing package discovery or check names."""

import os


def select_ci_tier(event: str, draft: bool, requested: str) -> str:
    if requested:
        if event != "workflow_dispatch" or requested not in ("daily", "merge", "all"):
            raise ValueError("A test tier can only be selected by manual workflow dispatch")
        return requested
    if event == "pull_request" and draft:
        return "daily"
    return "merge"


if __name__ == "__main__":
    print(select_ci_tier(
        os.environ["CI_TEST_EVENT"],
        os.environ.get("CI_TEST_DRAFT") == "true",
        os.environ.get("CI_TEST_TIER", ""),
    ))
