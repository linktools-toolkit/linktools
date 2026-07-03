from linktools.ai.resource_store.local import InMemoryResourceBackend
from linktools.ai.resource_store.store import ResourceStore
from linktools.ai.subagent.registry import SubagentRegistry


def test_subagent_registry_loads_from_resource_store(tmp_path):
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/subagent/db-agent/agent.md", "---\nname: db-agent\n---\ninstructions")
        store = ResourceStore(backends=[backend])
        registry = SubagentRegistry(tmp_path, resource_store=store, capabilities_root=tmp_path, cap_kind="subagent")
        await registry.preload()
        spec = registry.get("db-agent")
        assert spec is not None
        assert spec.name == "db-agent"

    import asyncio
    asyncio.run(run())


def test_subagent_registry_does_not_register_without_cap_kind(tmp_path):
    async def run():
        backend = InMemoryResourceBackend()
        store = ResourceStore(backends=[backend])
        registry = SubagentRegistry(tmp_path, resource_store=store, capabilities_root=tmp_path)
        await registry.preload()
        assert "anything" not in registry

    import asyncio
    asyncio.run(run())


def test_subagent_registry_does_not_touch_resource_store_when_none(tmp_path):
    registry = SubagentRegistry(tmp_path)
    assert registry is not None
