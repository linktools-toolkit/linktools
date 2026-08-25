#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_no_public_private_module_twins():
    duplicates = []
    for project in ROOT.iterdir():
        source_root = project / "src"
        if not project.name.startswith("linktools") or not source_root.is_dir():
            continue
        for private in source_root.rglob("_*.py"):
            if private.name.startswith("__"):
                continue
            public = private.with_name(private.name[1:])
            if public.is_file():
                duplicates.append(public.relative_to(ROOT).as_posix())

    assert not duplicates, "public/private module twins: " + ", ".join(sorted(duplicates))
