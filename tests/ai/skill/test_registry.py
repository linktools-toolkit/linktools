from linktools.ai.skill.registry import SkillRegistry


class _RecordingCapStore:
    def __init__(self) -> None:
        self.registered: "list[tuple[str, str]]" = []

    def register_primary(self, kind: str, primary_rel: str) -> None:
        self.registered.append((kind, primary_rel))

    async def iter_primaries(self, kind: str):
        return []


def test_skill_registry_registers_its_primary_filename_with_cap_store(tmp_path):
    cap_store = _RecordingCapStore()
    SkillRegistry(tmp_path, cap_store=cap_store)
    assert cap_store.registered == [("skill", "SKILL.md")]


def test_skill_registry_does_not_touch_cap_store_when_none(tmp_path):
    # No cap_store supplied -- constructing the registry must not raise.
    registry = SkillRegistry(tmp_path)
    assert registry is not None
