#!/usr/bin/env python3
from pathlib import Path

runtime = Path("linktools-ai/src/linktools/ai/runtime/_runtime_service.py")
text = runtime.read_text(encoding="utf-8").replace("from typing import Protocol, TypeGuard", "from typing import Protocol")
runtime.write_text(text, encoding="utf-8")

session = Path("linktools-ai/src/linktools/ai/runtime/_session.py")
text = session.read_text(encoding="utf-8")
text = text.replace("    ConversationHistoryRecord,\n", "")
text = text.replace("    HistoryQuality,\n", "")
session.write_text(text, encoding="utf-8")

executor = Path("linktools-ai/src/linktools/ai/agent/_executor.py")
text = executor.read_text(encoding="utf-8")
text = text.replace(
    "        for binding in definition.effective_capabilities:\n            materialized.extend(await binding.materialize(capability_context))",
    "        for capability in definition.effective_capabilities:\n            materialized.extend(await capability.materialize(capability_context))",
    1,
)
executor.write_text(text, encoding="utf-8")
print("lint blockers and binding shadow fixed")
