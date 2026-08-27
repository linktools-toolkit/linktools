#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate builtin compose snapshots used by cntr tests."""

import sys
import tempfile
from pathlib import Path

import _harness


SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "builtin"
BUILTIN_CONTAINERS = ["nginx", "lldap", "authelia", "safeline", "portainer", "flare"]


def main() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager = _harness.make_manager(root / "data", root / "temp")
        missing = [name for name in BUILTIN_CONTAINERS if name not in manager.containers]
        if missing:
            print("ERROR: expected builtin containers missing: %s" % missing, file=sys.stderr)
            return 2
        for name in BUILTIN_CONTAINERS:
            actual = _harness.normalize_compose(
                manager.containers[name].docker_compose,
                manager,
            )
            path = SNAPSHOT_DIR / (name + ".compose.json")
            path.write_text(actual, encoding="utf-8")
            print("WROTE: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
