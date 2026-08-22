#!/usr/bin/env python3
import subprocess

checks = (
    ("linktools-ai/src/linktools/ai/runtime/_execution.py", "_definition\\(|binding_snapshot|output_binding|binding_digest|AgentDefinition"),
    ("linktools-ai/src/linktools/ai/runtime/_local.py", "catalog\\.|binding_snapshot|output_binding|executor\\.execute|binding_digest"),
    ("linktools-ai/src/linktools/ai/runtime/_factory.py", "binding_snapshot|output_binding|compiler\\.restore|catalog\\.|RecoveryExecutionInput"),
    ("linktools-ai/src/linktools/ai/runtime/_subagent.py", "catalog\\.|AgentDefinition|binding_digest|planning|thinking"),
    ("linktools-ai/src/linktools/ai/runtime/_planner.py", "binding|AgentDefinition|catalog\\."),
    ("linktools-ai/src/linktools/ai/runtime/temporal", "LEGACY|CURRENT|binding_digest|agent_id|binding_snapshot|AgentBindingSnapshot"),
)
for file_name, pattern in checks:
    print(f"\n===== {file_name} :: {pattern} =====")
    subprocess.run(["grep", "-R", "-n", "-C", "2", "-E", pattern, file_name], check=False)
