#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pytest conftest for the external_adapter package.

Puts the package ``src`` directory on ``sys.path`` so the ``external_adapter``
package (src layout) is importable without an installed wheel -- the test
discovery path. When the package IS installed (isolated-venv wheel proof),
this sys.path insert is a harmless no-op (the installed package wins or the
src dir supplements it identically).

Also puts ``linktools-ai-testing/src`` on ``sys.path`` so the storage
conformance testkit (``linktools.ai.testing``, test-support code shipped from
the separate ``linktools-ai-testing`` wheel, never inside the core wheel) is
importable in-repo. This duplicates the same insert in the repo-root
``tests/conftest.py`` deliberately: this directory's own ``pyproject.toml``
makes pytest treat it as a conftest boundary in some multi-path invocations,
so ``tests/conftest.py`` is not reliably loaded when this package's tests run
alongside others in one session -- each conftest must be self-sufficient."""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_LINKTOOLS_AI_TESTING_SRC = (
    Path(__file__).parent.parent.parent / "linktools-ai-testing" / "src"
)
if str(_LINKTOOLS_AI_TESTING_SRC) not in sys.path:
    sys.path.insert(0, str(_LINKTOOLS_AI_TESTING_SRC))
