#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary exact patch driver for Python 3.10 verification."""

from pathlib import Path


ROOT = Path("linktools-ai/src/linktools/ai")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    path.write_text(text.replace(old, new))


def main() -> None:
    compat = ROOT / "_compat.py"
    if compat.exists():
        raise SystemExit(f"{compat}: compatibility module already exists")
    compat.write_text(
        '''#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Compatibility helpers for the supported Python baseline."""\n\ntry:\n    from enum import StrEnum\nexcept ImportError:\n    from enum import Enum\n\n    class StrEnum(str, Enum):\n        """Python 3.10-compatible subset of enum.StrEnum."""\n\n        def __str__(self) -> str:\n            return str(self.value)\n\n\n__all__ = ["StrEnum"]\n'''
    )

    imports = {
        ROOT / "errors.py": "from ._compat import StrEnum\n",
        ROOT / "core/_value.py": "from .._compat import StrEnum\n",
        ROOT / "core/_principal.py": "from .._compat import StrEnum\n",
        ROOT / "core/_redaction.py": "from .._compat import StrEnum\n",
        ROOT / "asset/_logical.py": "from .._compat import StrEnum\n",
        ROOT / "storage/_layer.py": "from .._compat import StrEnum\n",
        ROOT / "storage/_contracts.py": "from .._compat import StrEnum\n",
        ROOT / "storage/_dialects.py": "from .._compat import StrEnum\n",
        ROOT / "runtime/_tool.py": "from .._compat import StrEnum\n",
        ROOT / "runtime/_execution.py": "from .._compat import StrEnum\n",
        ROOT / "runtime/state/_plan.py": "from ..._compat import StrEnum\n",
        ROOT / "runtime/state/_root.py": "from ..._compat import StrEnum\n",
        ROOT / "runtime/state/_durability.py": "from ..._compat import StrEnum\n",
        ROOT / "runtime/state/_readmodel.py": "from ..._compat import StrEnum\n",
        ROOT / "runtime/state/_contracts.py": "from ..._compat import StrEnum\n",
        ROOT / "runtime/state/_steps.py": "from ..._compat import StrEnum\n",
        ROOT / "temporal/workflow/_execution.py": "from ..._compat import StrEnum\n",
    }
    for path, replacement in imports.items():
        replace_once(path, "from enum import StrEnum\n", replacement)

    test = Path("tests/ai/test_error_contract.py")
    text = test.read_text()
    anchor = '''def _map(error: Exception) -> AIError:\n    return _execution_error(\n        error,\n        usage_limits=UsageLimits(),\n        run_usage=RunUsage(),\n    )\n\n\n'''
    addition = '''def test_error_code_str_matches_wire_value() -> None:\n    assert str(ErrorCode.MODEL_API_ERROR) == ErrorCode.MODEL_API_ERROR.value\n\n\n'''
    if text.count(anchor) != 1:
        raise SystemExit(f"{test}: expected one test insertion anchor")
    test.write_text(text.replace(anchor, anchor + addition))


if __name__ == "__main__":
    main()
