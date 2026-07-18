#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.storage.file's __init__.py was an empty shell. This test
proves the shallow re-export resolves to the exact same object as the deep
submodule import, for each of the 7 in-scope File*Store classes."""

import pytest


@pytest.mark.parametrize(
    "name,submodule",
    [
        ("FileApprovalStore", "approval"),
        ("FileCheckpointStore", "checkpoint"),
        ("FileRunDefinitionStore", "definition"),
        ("FileEventStore", "event"),
        ("FileIdempotencyStore", "idempotency"),
        ("FileMemoryStore", "memory"),
        ("FileSwarmStore", "swarm"),
    ],
)
def test_file_store_reexport_identity(name, submodule):
    import importlib

    shallow_mod = importlib.import_module("linktools.ai.storage.file")
    deep_mod = importlib.import_module(f"linktools.ai.storage.file.{submodule}")
    assert getattr(shallow_mod, name) is getattr(deep_mod, name)
