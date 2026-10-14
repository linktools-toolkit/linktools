#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider code is loaded when a model operation needs it."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_custom_model_startup_defers_openai_until_provider_operations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent('''
        import sys

        from pydantic_ai.messages import BinaryContent
        from pydantic_ai.models.test import TestModel
        from linktools.ai.model import ModelRegistry, model_binding_error, validate_model_inputs
        from linktools.ai.runtime import Runtime

        class OfflineBinding:
            route_id = "offline"
            provider = "test"
            model_identity = "test:offline"
            vision = False
            contract = {"provider": provider, "model_identity": model_identity}

            def materialize(self) -> TestModel:
                return TestModel(custom_output_text="offline answer")

        registry = ModelRegistry()
        registry.register(OfflineBinding())
        binding = registry.capture().resolve("offline")
        assert isinstance(binding.materialize(), TestModel)
        decisions = validate_model_inputs(
            binding, BinaryContent(b"offline file", media_type="application/x-offline")
        )
        assert decisions[0].allowed and decisions[0].transport is None
        assert not any(name == "openai" or name.startswith("openai.") for name in sys.modules)

        assert model_binding_error(RuntimeError("unclassified")) is None
        registry.register_openai("selected", model="gpt-test", api_key="unused")
        selected = registry.capture().resolve("selected")
        assert selected.model_identity == "openai:gpt-test"
        assert dict(selected.contract) == {
            "provider": "openai",
            "model_identity": "openai:gpt-test",
            "vision": False,
            "settings": {},
        }
    ''')
    environment = {
        **os.environ,
        "LINKTOOLS_PATH": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((
            str(root / "linktools-ai" / "src"),
            str(root / "linktools" / "src"),
            os.environ.get("PYTHONPATH", ""),
        )),
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
