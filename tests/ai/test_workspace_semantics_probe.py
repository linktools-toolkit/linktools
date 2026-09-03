#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pytest
from linktools.ai.capability import workspace_tool_contributions
from linktools.ai.workspace import Workspace


def test_workspace_semantics_probe(tmp_path: Path) -> None:
    semantics = {
        contribution.id: contribution.semantic_contract
        for contribution in workspace_tool_contributions(Workspace.load(tmp_path))
    }
    print("WORKSPACE_SEMANTICS_START")
    print(json.dumps(semantics, ensure_ascii=False, indent=2, sort_keys=True))
    print("WORKSPACE_SEMANTICS_END")
    pytest.fail("capture workspace semantic baseline")
