#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.cwd()
for base in (ROOT / "linktools-ai" / "src", ROOT / "tests" / "ai"):
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        updated = text.replace("AgentDefinitionCatalog", "AgentCatalog")
        if updated != text:
            path.write_text(updated, encoding="utf-8")
