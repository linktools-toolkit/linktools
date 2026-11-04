# MCP protocol tools and ownership

Normal applications continue to register `MCPServerSpec` through
`RuntimeFeatures.mcp(...)` and select tools on the Agent. Runtime prepares the
transport, applies stable model names, authorization, budgets and fixtures, and
performs final cleanup. No additional application object or context is required.

## Advanced SDK integration

`linktools.ai.capability.mcp.MCPToolset` implements the existing PydanticAI
Toolset protocol. Its constructor is:

```python
MCPToolset(transport: ClientTransport, *, server: MCPServerSpec)
```

The transport is already configured by its caller. The server supplies identity,
transport-kind diagnostics and initialization/read timeouts; this constructor
does not deploy server resources or reinterpret its command/URL. The toolset
owns its client, but not the caller's Workspace, AssetStore or other services.

It discovers tools, rejects duplicate remote names before dictionary conversion,
and supplies model-correctable argument feedback for supported JSON schemas.
Unknown schema dialects and external references stay with the server. Connection
failures are sanitized; cancellation and unknown tool outcomes retain their
existing meaning. It does not add Runtime authorization, model-name mapping,
fixtures, effect accounting, or an alternative retry mechanism.

SDK entry and reference counting are retained. Exit cleanup failures are saved
instead of replacing an active execution error; final cleanup reports them.
The owner must additionally await
`close_resources() -> None` in final cleanup, even when entry or exit failed.
This closes the client and reports any recorded exit failure. Cancellation of
this direct call propagates, with an earlier exit failure retained as its cause;
it does not poison a later cleanup retry. A caller with an
existing primary failure should preserve that failure when cleanup also fails.
Runtime already performs this operation under cancellation shielding before
Workspace process-tree cleanup. Native callers retain responsibility for their
own cancellation and other resource lifetimes.

This complete standalone example uses a real local stdio server and an offline
model; it does not require credentials or a model service:

```python
import asyncio
import sys

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from linktools.ai.capability.mcp import MCPToolset
from linktools.ai.spec import MCPServerSpec

SERVER = '''
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "echo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Echo a value",
                  "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}},
                                  "required": ["value"]}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}],
                  "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''

async def main() -> None:
    server = MCPServerSpec("echo", sys.executable, ("-c", SERVER))
    tools = MCPToolset(StdioTransport(sys.executable, args=["-c", SERVER]), server=server)
    agent = Agent(TestModel(call_tools="all", custom_output_text="done"), toolsets=[tools])
    primary = None
    try:
        async with agent:
            result = await agent.run("Echo a value")
            assert result.output == "done"
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            await tools.close_resources()
        except BaseException as cleanup:
            if primary is not None:
                raise primary from cleanup
            raise

asyncio.run(main())
```

The plain SDK `pydantic_ai.mcp.MCPToolset` remains available and unchanged.
Runtime fixtures never construct this protocol client; they retain their frozen
recorded definitions and responses. Current default replay safety and retained
binding policy remain execution-side concerns. No stored format, selector,
source revision or model-visible tool name changes with this ownership split.
