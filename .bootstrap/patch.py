#!/usr/bin/env python3
import subprocess

PATTERNS = (
    r"AgentDefinitionCatalog",
    r"SessionRecord\(",
    r"SessionView\(",
    r"ContextProjection\(",
    r"\.binding_digest",
    r"binding_digest=",
    r"AgentDefinition\.output_binding|\.output_binding",
    r"AgentDefinition\.binding_snapshot|\.binding_snapshot",
    r"_AGENT_TASK_LEGACY_V1_FIELDS|_AGENT_TASK_CURRENT_V1_FIELDS",
    r"_EXECUTION_LEGACY_V1_FIELDS|_EXECUTION_CURRENT_V1_FIELDS",
    r"AssetTypeRegistry|AssetTypeRegistrySnapshot",
    r"runtime\.agent\([^\n]*(output|planning|thinking)=",
)

for pattern in PATTERNS:
    print(f"\n===== RG {pattern} =====")
    subprocess.run(
        [
            "rg",
            "-n",
            "--glob",
            "*.py",
            "--glob",
            "*.json",
            pattern,
            "linktools-ai/src",
            "tests/ai",
        ],
        check=False,
    )
