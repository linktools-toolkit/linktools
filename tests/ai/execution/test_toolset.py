import asyncio

from linktools.ai.execution.toolset import BuiltinToolContext, build_builtin_toolset


class _FakeBackend:
    def __init__(self):
        self.last_diff = None

    async def apply_patch(self, diff):
        self.last_diff = diff
        return {"ok": True, "output": "patched"}


def test_apply_patch_tool_registered_when_file_enabled():
    backend = _FakeBackend()
    toolset = build_builtin_toolset(
        BuiltinToolContext(backend=backend, enabled_tools={"file"})
    )
    # FunctionToolset doesn't expose a stable public listing API across versions;
    # the reliable check is invoking the tool through the toolset's function map.
    assert "apply_patch" in toolset.tools


def test_apply_patch_tool_forwards_to_backend():
    backend = _FakeBackend()
    toolset = build_builtin_toolset(
        BuiltinToolContext(backend=backend, enabled_tools={"file"})
    )
    fn = toolset.tools["apply_patch"].function
    result = asyncio.run(fn("--- a\n+++ b\n"))
    assert result == {"ok": True, "output": "patched"}
    assert backend.last_diff == "--- a\n+++ b\n"


def test_apply_patch_tool_absent_when_file_disabled():
    backend = _FakeBackend()
    toolset = build_builtin_toolset(
        BuiltinToolContext(backend=backend, enabled_tools=set())
    )
    assert "apply_patch" not in toolset.tools
