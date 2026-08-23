from pathlib import Path

capabilities = Path("linktools-ai/src/linktools/ai/agent/_capabilities.py")
text = capabilities.read_text()

anchor = '''def tool_allowed_in_planning(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
    trusted_mcp_selectors: "tuple[str, ...]",
) -> bool:
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class in {"control", "filesystem.read", "memory.read"}:
        if tool_class == "control":
            return tool_def.name == "write_plan" and tool_def.capability_id == _PLANNING_CAPABILITY_ID
        expected_capability = (
            _WORKSPACE_FILESYSTEM_CAPABILITY_ID if tool_class == "filesystem.read" else _MEMORY_CAPABILITY_ID
        )
        return tool_def.capability_id == expected_capability
    if tool_class is not None:
        return False
    if tool_def.capability_id in trusted_mcp_selectors:
        return bool(tool_def.metadata.get(PLAN_SAFE_METADATA_KEY, False))
    metadata = tool_def.metadata.get(PLAN_SAFE_METADATA_KEY)
    if metadata is None:
        return False
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return metadata
'''
replacement = '''def tool_name_allowed(name: str, allow_tools: "tuple[str, ...]") -> bool:
    return "*" in allow_tools or name in allow_tools


def _trusted_tool_capability(name: str, tool_class: str) -> "str | None":
    if tool_class == "control":
        if name in SKILL_TOOL_NAMES:
            return _SKILL_CAPABILITY_ID
        if name in PLANNING_TOOL_NAMES:
            return _PLANNING_CAPABILITY_ID
        if name in SUBAGENT_TOOL_NAMES:
            return _SUBAGENT_CAPABILITY_ID
        return None
    if tool_class in {"filesystem.read", "filesystem.write"}:
        if name not in WORKSPACE_FILESYSTEM_TOOL_NAMES:
            return None
        is_read = name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        if is_read != (tool_class == "filesystem.read"):
            return None
        return _WORKSPACE_FILESYSTEM_CAPABILITY_ID
    if tool_class == "shell":
        return _WORKSPACE_SHELL_CAPABILITY_ID if name in WORKSPACE_SHELL_TOOL_NAMES else None
    if tool_class in {"memory.read", "memory.write"}:
        if name not in MEMORY_TOOL_NAMES:
            return None
        is_read = name in MEMORY_READ_TOOL_NAMES
        if is_read != (tool_class == "memory.read"):
            return None
        return _MEMORY_CAPABILITY_ID
    return None


def tool_is_control(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
) -> bool:
    _validate_trusted_tool_classes(trusted_tool_classes)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class != "control":
        return False
    expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
    return expected_capability is not None and tool_def.capability_id == expected_capability


def tool_allowed_in_planning(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
    trusted_mcp_selectors: "tuple[str, ...]",
) -> bool:
    _validate_trusted_tool_classes(trusted_tool_classes)
    _validate_trusted_mcp_selectors(trusted_mcp_selectors)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class is not None:
        expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
        if expected_capability is None:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        if tool_def.capability_id == expected_capability:
            return tool_class in {"control", "filesystem.read", "memory.read"}
    if tool_def.capability_id in trusted_mcp_selectors:
        if not tool_def.name.startswith(f"{tool_def.capability_id}__"):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        return False
    if any(tool_def.name.startswith(f"{selector}__") for selector in trusted_mcp_selectors):
        return False
    metadata = (tool_def.metadata or {}).get(PLAN_SAFE_METADATA_KEY)
    if metadata is None:
        return False
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return metadata
'''
if text.count(anchor) != 1:
    raise SystemExit(f"planning policy block count={text.count(anchor)}")
text = text.replace(anchor, replacement)

old_filter = "    names.intersection_update(allow_tools)\n"
new_filter = "    names = {name for name in names if tool_name_allowed(name, allow_tools)}\n"
if text.count(old_filter) != 1:
    raise SystemExit(f"platform allow filter count={text.count(old_filter)}")
text = text.replace(old_filter, new_filter)

old_exports = '''    "select_platform_tool_names",
    "tool_allowed_in_planning",
]'''
new_exports = '''    "select_platform_tool_names",
    "tool_allowed_in_planning",
    "tool_is_control",
    "tool_name_allowed",
]'''
if text.count(old_exports) != 1:
    raise SystemExit(f"capability export block count={text.count(old_exports)}")
text = text.replace(old_exports, new_exports)
capabilities.write_text(text)

tests = Path("tests/ai/test_spec_capability_refactor.py")
test_text = tests.read_text()
old_import = "from linktools.ai.agent import select_platform_tool_names\n"
new_import = "from linktools.ai.agent import select_platform_tool_names, tool_name_allowed\n"
if test_text.count(old_import) != 1:
    raise SystemExit(f"agent import count={test_text.count(old_import)}")
test_text = test_text.replace(old_import, new_import)

marker = '''def test_planning_gate_requires_framework_filesystem_provenance() -> None:
'''
addition = '''def test_platform_tool_selection_honors_wildcard_allow_tools() -> None:
    assert tool_name_allowed("read_memory", ("*",))
    assert select_platform_tool_names(
        allow_tools=("*",),
        memory_scope="memory",
    ) == ("delete_memory", "read_memory", "search_memory", "write_memory")


'''
if test_text.count(marker) != 1:
    raise SystemExit(f"test marker count={test_text.count(marker)}")
test_text = test_text.replace(marker, addition + marker)
tests.write_text(test_text)
