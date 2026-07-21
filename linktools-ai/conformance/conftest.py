#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Makes the storage conformance testkit (``linktools-ai/testing/``, a sibling
of this package, never packaged into the wheel) importable as ``testing`` when
this package is run in place. When copied into an isolated venv alongside a
sibling ``testing/`` dir (the wheel-only gold-standard test does this), the
same relative layout resolves the same way."""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
