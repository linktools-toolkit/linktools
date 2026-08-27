#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model binding must not expose credential material."""

from linktools.ai.model import ModelRegistry


def test_model_binding_fingerprint_excludes_secret_material() -> None:
    registry = ModelRegistry.openai(model="gpt-test", api_key="secret")
    binding = registry.snapshot().resolve("default")

    assert "secret" not in repr(binding)
    assert "secret" not in binding.fingerprint
    assert binding.model_identity == "openai:gpt-test"
