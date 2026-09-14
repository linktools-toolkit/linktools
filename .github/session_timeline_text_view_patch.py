from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


source = "linktools-ai/src/linktools/ai/runtime/_input.py"
replace_once(
    source,
    '''        view = (\n            dict(value.view)\n            if isinstance(value, _MaterializedUserContent)\n            else _input_view(value)\n        )\n        canonical = validate_user_input(value)\n        if isinstance(canonical, str):\n            return StoredUserInput(\n                _TEXT_CODEC,\n                StoredPayload.inline_text(canonical),\n                view,\n            )\n        payload = StoredPayload.inline_json(_encode_user_content(canonical))\n''',
    '''        canonical = validate_user_input(value)\n        if isinstance(canonical, str):\n            return StoredUserInput(\n                _TEXT_CODEC,\n                StoredPayload.inline_text(canonical),\n            )\n        view = (\n            dict(value.view)\n            if isinstance(value, _MaterializedUserContent)\n            else _input_view(value)\n        )\n        payload = StoredPayload.inline_json(_encode_user_content(canonical))\n''',
)

test = "tests/ai/test_file_input_regressions.py"
replace_once(
    test,
    '''from linktools.ai.runtime._input import ExecutionInputMaterializer\n''',
    '''from linktools.ai.runtime._input import (\n    ExecutionInputMaterializer,\n    stored_user_input_view,\n)\n''',
)
replace_once(
    test,
    '''        assert stored.view == {\n            "version": 1,\n            "prompt": {"kind": "text", "text": "plain text"},\n            "files": [],\n        }\n''',
    '''        assert stored.view is None\n        assert stored_user_input_view(stored) == {\n            "version": 1,\n            "prompt": {"kind": "text", "text": "plain text"},\n            "files": [],\n        }\n''',
)

print("plain text display view deduplicated")
