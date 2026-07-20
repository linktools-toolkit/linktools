#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9 op 4 / op 5: run the public conformance testkit against a from-
scratch EXTERNAL adapter that imports ONLY public ``linktools.ai`` paths.

This proves two things the plan §9 acceptance criteria depend on:

1. The storage Protocols are SUFFICIENT -- a conformant adapter can be built
   against ``linktools.ai.storage.protocols`` + ``linktools.ai.artifact.models``
   alone, with no private/core imports (op 4). The import-surface guard below
   enforces this mechanically.
2. The conformance testkit is BACKEND-AGNOSTIC -- the same contracts the
   in-repo reference backends pass (``test_conformance_testkit.py``) also pass
   against this independent in-memory implementation (op 5 / op 7 reuse).

Plan §9 failure handling: if this adapter ever NEEDS a private import to
conform, the Protocol design is inadequate and the work returns to Phase 3."""

import ast
import pathlib

from linktools.ai.storage.testing import (
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
    LeaseCoordinatorContract,
)

from .example_external_adapter import (
    InMemoryArtifactBlobStore,
    InMemoryArtifactRecordStore,
    InMemoryLeaseCoordinator,
)

# The allowlist of public modules an EXTERNAL adapter may import. Anything
# outside this set -- underscore-prefixed modules, ``_runtime``, the in-repo
# reference backends under ``storage.filesystem`` / ``storage.sqlalchemy`` /
# ``storage.coordination`` -- would defeat the point: the adapter exists to
# prove the PUBLIC Protocols suffice.
_PUBLIC_ADAPTER_IMPORTS = frozenset(
    {
        "linktools.ai.storage.protocols",
        "linktools.ai.storage.testing",
        "linktools.ai.artifact.models",
    }
)


def test_external_adapter_imports_only_public_paths() -> None:
    src = pathlib.Path(__file__).with_name("example_external_adapter.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    imported: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported.add(node.module)
    linktools_imports = {m for m in imported if m.startswith("linktools")}
    non_public = linktools_imports - _PUBLIC_ADAPTER_IMPORTS
    assert not non_public, (
        "external adapter must import only public Protocol/testkit paths; "
        f"found non-public linktools imports: {sorted(non_public)}"
    )


class TestExternalBlobStoreConformance(ArtifactBlobStoreContract):
    def blob_store(self):
        return InMemoryArtifactBlobStore()


class TestExternalRecordStoreConformance(ArtifactRecordStoreContract):
    def record_store(self):
        return InMemoryArtifactRecordStore()


class TestExternalLeaseCoordinatorConformance(LeaseCoordinatorContract):
    def coordinator(self):
        return InMemoryLeaseCoordinator()
