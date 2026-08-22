#!/usr/bin/env python3
import subprocess

PATTERNS = (
    "AgentDefinitionCatalog",
    "SessionRecord\\(",
    "SessionView\\(",
    "ContextProjection\\(",
    "\\.binding_digest",
    "binding_digest=",
    "\\.output_binding",
    "\\.binding_snapshot",
    "_AGENT_TASK_LEGACY_V1_FIELDS|_AGENT_TASK_CURRENT_V1_FIELDS",
    "_EXECUTION_LEGACY_V1_FIELDS|_EXECUTION_CURRENT_V1_FIELDS",
    "AssetTypeRegistry|AssetTypeRegistrySnapshot",
    "runtime\\.agent\\(.*(output|planning|thinking)=",
)

for pattern in PATTERNS:
    print(f"\n===== GREP {pattern} =====")
    subprocess.run(
        ["grep", "-R", "-n", "-E", "--include=*.py", "--include=*.json", pattern, "linktools-ai/src", "tests/ai"],
        check=False,
    )
