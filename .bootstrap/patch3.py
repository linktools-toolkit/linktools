#!/usr/bin/env python3
from pathlib import Path

path = Path("tests/ai/test_agent_composition.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "import pytest\nfrom pydantic_ai.capabilities import AbstractCapability\n",
    "import pytest\nfrom pydantic import BaseModel\nfrom pydantic_ai.capabilities import AbstractCapability\n",
    1,
)
marker = '''class _DurableCapability(AbstractCapability[None]):\n'''
model = '''class CompositionStructuredOutput(BaseModel):\n    value: str\n\n\n'''
if model not in text:
    if marker not in text:
        raise SystemExit("durable capability marker not found")
    text = text.replace(marker, model + marker, 1)
text = text.replace(
    '''async def test_agent_and_binding_identity_split_acceptance(tmp_path) -> None:\n    from pydantic import BaseModel\n    from linktools.ai.model import ModelRegistry\n''',
    '''async def test_agent_and_binding_identity_split_acceptance(tmp_path) -> None:\n    from linktools.ai.model import ModelRegistry\n''',
    1,
)
text = text.replace("            output=BaseModel,\n", "            output=CompositionStructuredOutput,\n", 1)
if "output=BaseModel" in text:
    raise SystemExit("abstract BaseModel output residue remains")
path.write_text(text, encoding="utf-8")
print("identity acceptance test now uses an importable concrete output model")
