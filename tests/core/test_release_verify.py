#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from scripts import verify as release_verify


def _artifact(path: Path, name: str) -> release_verify._Artifact:
    return release_verify._Artifact(
        path,
        "wheel",
        name,
        "0.10.0",
        ">=3.6",
    )


def test_candidate_constraints_pin_other_repository_wheels(tmp_path: Path) -> None:
    wheels = {
        "linktools": _artifact(tmp_path / "linktools.whl", "linktools"),
        "linktools-common": _artifact(
            tmp_path / "linktools_common.whl",
            "linktools-common",
        ),
    }

    constraints = release_verify._candidate_constraints(
        wheels,
        exclude="linktools-common",
    )

    assert constraints == (
        "linktools @ %s" % (tmp_path / "linktools.whl").as_uri(),
    )
