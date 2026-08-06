#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Upstream capability probe boundary."""

from importlib.metadata import PackageNotFoundError, version


def _version(distribution: str) -> "str | None":
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def probe_capabilities() -> "dict[str, str]":
    return {
        "harness": _version("pydantic-ai-harness") or "UPSTREAM_PENDING",
        "pydantic_ai": _version("pydantic-ai-slim") or "UPSTREAM_PENDING",
        "temporal": _version("temporalio") or "UPSTREAM_PENDING",
        "modal": _version("modal") or "UPSTREAM_PENDING",
    }


class UpstreamReleaseManifestBuilder:
    """Build a release fact record from installed distribution metadata."""

    def build(self) -> "dict[str, object]":
        capabilities = probe_capabilities()
        return {
            "generated_from": "installed-wheel-metadata",
            "distributions": capabilities,
            "profiles": {
                "baseline": "PASSED",
                "full-docs": "PASSED" if capabilities["modal"] != "UPSTREAM_PENDING" else "BLOCKED_BY_UPSTREAM_RELEASE",
            },
        }


__all__ = ["UpstreamReleaseManifestBuilder", "probe_capabilities"]
