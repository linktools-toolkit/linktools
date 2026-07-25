#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Makes the storage conformance testkit importable as
``linktools.ai.testing`` when this package is run in place against the source
tree. The testkit ships from the separate ``linktools-ai-testing`` distribution
; in-repo it resolves via its src dir on sys.path, and in the
isolated-venv gold-standard run it resolves via the installed testing wheel
(this insert is then a harmless no-op). The ``linktools.ai`` namespace is
stitched across both wheels by ``pkgutil.extend_path`` in linktools-ai's
``__init__.py``."""

import sys
from pathlib import Path

_TESTING_SRC = (
    Path(__file__).resolve().parent.parent.parent
    / "linktools-ai-testing"
    / "src"
)
if _TESTING_SRC.is_dir() and str(_TESTING_SRC) not in sys.path:
    sys.path.insert(0, str(_TESTING_SRC))
