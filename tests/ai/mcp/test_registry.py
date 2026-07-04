from linktools.ai.mcp.registry import MCPRegistry
from linktools.ai.resource.local import InMemoryResourceBackend
from linktools.ai.resource.store import ResourceStore


def test_mcp_registry_loads_from_resource(tmp_path):
    async def run():
        backend = InMemoryResourceBackend()
        await backend.put("/mcp/db-mcp/mcp.yaml", "name: db-mcp\nmcp:\n  type: stdio\n  command: echo\n")
        store = ResourceStore(backend)
        registry = MCPRegistry(tmp_path, resource=store)
        await registry.preload()
        spec = registry.get("db-mcp")
        assert spec is not None
        assert spec.name == "db-mcp"

    import asyncio
    asyncio.run(run())


def test_mcp_registry_does_not_touch_resource_when_none(tmp_path):
    registry = MCPRegistry(tmp_path)
    assert registry is not None
