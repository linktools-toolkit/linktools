#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate / check linktools-cntr builtin compose snapshots.

Usage:
    python scripts/cntr_generate_snapshots.py          # (re)write snapshots
    python scripts/cntr_generate_snapshots.py --check   # fail if drift (CI)

Snapshots are the ``normalize_compose`` output of each builtin container's
``docker_compose``. They lock the current compose ABI so internal changes to
cntr can be verified not to alter generated output.
"""
import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests" / "cntr"))

import _harness  # noqa: E402

SNAPSHOT_DIR = REPO_ROOT / "tests" / "cntr" / "snapshots" / "builtin"
BUILTIN_CONTAINERS = ["nginx", "lldap", "authelia", "safeline", "portainer", "flare"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare rendered output to committed snapshots; "
                             "exit non-zero on any mismatch or missing snapshot")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        manager = _harness.make_manager(Path(tmp) / "data", Path(tmp) / "temp")

        missing = [n for n in BUILTIN_CONTAINERS if n not in manager.containers]
        if missing:
            print(f"ERROR: expected builtin containers missing: {missing}", file=sys.stderr)
            return 2

        if not args.check:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        rc = 0
        for name in BUILTIN_CONTAINERS:
            actual = _harness.normalize_compose(manager.containers[name].docker_compose, manager)
            path = SNAPSHOT_DIR / f"{name}.compose.json"
            if args.check:
                if not path.exists():
                    print(f"MISSING snapshot: {path}")
                    rc = 1
                    continue
                expected = path.read_text(encoding="utf-8")
                if expected != actual:
                    print(f"MISMATCH: {name} (see {path})")
                    rc = 1
                else:
                    print(f"OK: {name}")
            else:
                path.write_text(actual, encoding="utf-8")
                print(f"WROTE: {path}")
        return rc


if __name__ == "__main__":
    sys.exit(main())
