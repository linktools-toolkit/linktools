#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from scripts import verify as release_verify


def _artifact(name: str, *requires_dist: str) -> release_verify._Artifact:
    return release_verify._Artifact(
        Path("/tmp/%s.whl" % name),
        "wheel",
        name,
        "0.10.0",
        ">=3.6",
        requires_dist,
    )


def test_candidate_install_order_uses_repository_dependencies_only() -> None:
    artifacts = {
        "linktools": _artifact("linktools", "filelock>=3.4.0"),
        "linktools-common": _artifact(
            "linktools-common",
            "linktools[cli]>=0.10.0",
            "lief>0.10.1; extra == 'lief'",
        ),
    }

    order = release_verify._candidate_install_order(
        "linktools-common",
        artifacts,
        {"linktools", "linktools-common", "linktools-mobile"},
    )

    assert order == ("linktools", "linktools-common")


def test_candidate_install_order_rejects_missing_repository_dependency() -> None:
    artifacts = {
        "linktools-common": _artifact(
            "linktools-common",
            "linktools[cli]>=0.10.0",
        ),
    }

    with pytest.raises(ValueError, match="requires repository candidate linktools"):
        release_verify._candidate_install_order(
            "linktools-common",
            artifacts,
            {"linktools", "linktools-common"},
        )
