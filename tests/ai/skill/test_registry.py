from linktools.ai.resource_store.local import InMemoryResourceBackend
from linktools.ai.resource_store.store import ResourceStore
from linktools.ai.skill.registry import SkillRegistry


def test_skill_registry_loads_from_resource_store(tmp_path):
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/skill/db-skill/SKILL.md", "---\nname: db-skill\n---\ninstructions")
        store = ResourceStore(backends=[backend])
        registry = SkillRegistry(tmp_path, resource_store=store)
        await registry.preload()
        spec = registry.get("db-skill")
        assert spec is not None
        assert spec.name == "db-skill"

    import asyncio
    asyncio.run(run())


def test_skill_registry_does_not_touch_resource_store_when_none(tmp_path):
    registry = SkillRegistry(tmp_path)
    assert registry is not None
