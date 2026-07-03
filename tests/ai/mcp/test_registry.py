from linktools.ai.mcp.registry import MCPRegistry


class _RecordingCapStore:
    def __init__(self) -> None:
        self.registered: "list[tuple[str, str]]" = []

    def register_primary(self, kind: str, primary_rel: str) -> None:
        self.registered.append((kind, primary_rel))

    async def iter_primaries(self, kind: str):
        return []


def test_mcp_registry_registers_its_primary_filename_with_cap_store(tmp_path):
    cap_store = _RecordingCapStore()
    MCPRegistry(tmp_path, cap_store=cap_store)
    assert cap_store.registered == [("mcp", "mcp.yaml")]


def test_mcp_registry_does_not_touch_cap_store_when_none(tmp_path):
    registry = MCPRegistry(tmp_path)
    assert registry is not None
