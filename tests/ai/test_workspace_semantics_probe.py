#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import zlib
from pathlib import Path

import pytest
from linktools.ai.capability import workspace_tool_contributions
from linktools.ai.workspace import Workspace


def test_workspace_semantics_probe(tmp_path: Path) -> None:
    semantics = {
        contribution.id: contribution.semantic_contract
        for contribution in workspace_tool_contributions(Workspace.load(tmp_path))
    }
    raw = json.dumps(
        semantics,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    print("WORKSPACE_SEMANTICS_ZLIB_BASE64=" + base64.b64encode(zlib.compress(raw, 9)).decode("ascii"))
    pytest.fail("capture compact workspace semantic baseline")
