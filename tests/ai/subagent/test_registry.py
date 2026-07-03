from linktools.ai.subagent.registry import SubagentRegistry


class _RecordingCapStore:
    def __init__(self) -> None:
        self.registered: "list[tuple[str, str]]" = []

    def register_primary(self, kind: str, primary_rel: str) -> None:
        self.registered.append((kind, primary_rel))

    async def iter_primaries(self, kind: str):
        return []


def test_subagent_registry_registers_its_primary_filename_with_cap_store(tmp_path):
    cap_store = _RecordingCapStore()
    SubagentRegistry(tmp_path, cap_store=cap_store, capabilities_root=tmp_path, cap_kind="subagent")
    assert cap_store.registered == [("subagent", "agent.md")]


def test_markdown_agent_registry_does_not_register_without_cap_kind(tmp_path):
    # cap_store given but cap_kind omitted -- MarkdownAgentRegistry has no kind to register under.
    cap_store = _RecordingCapStore()
    SubagentRegistry(tmp_path, cap_store=cap_store, capabilities_root=tmp_path)
    assert cap_store.registered == []


def test_subagent_registry_does_not_touch_cap_store_when_none(tmp_path):
    registry = SubagentRegistry(tmp_path)
    assert registry is not None
