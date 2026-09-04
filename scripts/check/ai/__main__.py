#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from .architecture import ArchitecturePolicyChecker


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "linktools-ai" / "src" / "linktools" / "ai"
LINKTOOLS_SOURCE_ROOT = ROOT / "linktools" / "src" / "linktools"


def main() -> int:
    result = ArchitecturePolicyChecker().check(
        SOURCE_ROOT,
        external_roots=(LINKTOOLS_SOURCE_ROOT,),
    )
    if result.passed:
        print("[+] linktools-ai architecture gate passed")
        return 0
    for error in result.errors:
        print("[-] %s" % error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
