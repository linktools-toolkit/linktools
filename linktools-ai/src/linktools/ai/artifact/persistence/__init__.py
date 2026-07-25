#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact persistence: the ArtifactRecordStore Protocol (the artifact
domain's own access-control + provenance fact source) plus the in-repo
reference adapters. Concrete adapters live here, not in storage/, so storage
never imports the artifact domain."""

from .protocols import ArtifactRecordStore

__all__: "list[str]" = ["ArtifactRecordStore"]
