#!/usr/bin/env python3
import subprocess

checks = (
    ("linktools-ai/src/linktools/ai/runtime/state/_codec.py", "SessionRecord|ContextProjection|binding_digest"),
    ("linktools-ai/src/linktools/ai/runtime/state/_steps.py", "ContextProjection|projection\\.binding_digest|binding_digest.*projection|_projection_bindings"),
    ("linktools-ai/src/linktools/ai/runtime/state/_history.py", "ContextProjection|projection\\.binding_digest|binding_digest"),
    ("linktools-ai/src/linktools/ai/runtime/state/_contracts.py", "class RecoveryExecutionInput|class ContextProjection|class SessionRecord|class ExecutionRecord|binding_digest"),
    ("linktools-ai/src/linktools/ai/runtime/_session.py", "binding_digest"),
    ("linktools-ai/src/linktools/ai/runtime/service_api.py", "SessionView|class SessionService|binding_digest"),
)
for file_name, pattern in checks:
    print(f"\n===== {file_name} :: {pattern} =====")
    subprocess.run(["grep", "-n", "-C", "3", "-E", pattern, file_name], check=False)
raise RuntimeError("diagnostic only")
