import asyncio

from linktools.ai.core.runtime import AgentKernel


class _Registry:
    def __init__(self, specs):
        self._specs = {spec.name: spec for spec in specs}

    def all(self):
        return list(self._specs.values())

    def __contains__(self, item):
        return item in self._specs

    def get(self, item):
        return self._specs[item]

    def resolve_by_capability(self, item):
        return None


def _kernel() -> AgentKernel:
    return AgentKernel(
        skill_registry=_Registry([]),
        subagent_registry=_Registry([]),
        mcp_registry=_Registry([]),
    )


def test_agent_kernel_is_an_event_handler_mixin():
    kernel = _kernel()
    events = []
    kernel.on("agent_start", lambda **kwargs: events.append(("agent_start", kwargs)))
    kernel.trigger("agent_start", trace_id="t1", agent_id="a1")
    assert events == [("agent_start", {"trace_id": "t1", "agent_id": "a1"})]


def test_agent_kernel_trigger_swallows_handler_exceptions():
    kernel = _kernel()

    def _boom(**kwargs):
        raise ValueError("handler blew up")

    kernel.on("agent_start", _boom)
    # Must not raise -- EventHandlerMixin.trigger catches and logs per-handler,
    # matching HookRegistry.fire's existing behavior.
    kernel.trigger("agent_start", trace_id="t1")


def test_subagent_capability_fires_subagent_events_not_mcp_events():
    from linktools.ai.subagent.capability import SubagentCapability

    kernel = _kernel()
    fired = []
    kernel.on("subagent_start", lambda **kwargs: fired.append(("subagent_start", kwargs)))
    kernel.on("subagent_end", lambda **kwargs: fired.append(("subagent_end", kwargs)))
    kernel.on("mcp_call_start", lambda **kwargs: fired.append(("mcp_call_start", kwargs)))
    kernel.on("post_mcp_call", lambda **kwargs: fired.append(("post_mcp_call", kwargs)))

    async def run_subagent_fn(subagent_id, input, call_id):
        return {"ok": True}

    cap = SubagentCapability(
        run_subagent_fn=run_subagent_fn,
        allowed_subagents={"child"},
        kernel=kernel,
        trace_id="t1",
        parent_call_id=None,
    )

    class _FakeCall:
        tool_call_id = "call-1"

    class _FakeToolDef:
        name = "call_subagent"

    async def handler(args):
        return await run_subagent_fn(args["subagent_id"], args.get("input"), call_id="call-1")

    asyncio.run(cap.wrap_tool_execute(
        None, call=_FakeCall(), tool_def=_FakeToolDef(),
        args={"subagent_id": "child", "input": {}}, handler=handler,
    ))

    fired_events = [name for name, _ in fired]
    assert fired_events == ["subagent_start", "subagent_end"]
    start_kwargs = fired[0][1]
    assert "subagent_id" in start_kwargs or "parent_agent_id" in start_kwargs
    assert "server" not in start_kwargs
    assert "tool_name" not in start_kwargs


def test_build_context_stores_caller_supplied_context_dict():
    from linktools.ai.core.registry import AgentSpec

    kernel = _kernel()
    spec = AgentSpec(name="a", path=None, base_dir=None, enabled=True, model="standard")

    class _FakeSession:
        pass

    ctx = kernel.build_context(
        spec, _FakeSession(), builtin_tool_names=frozenset(), context={"trace_id": "T1", "tenant": "acme"},
    )

    assert ctx.context == {"trace_id": "T1", "tenant": "acme"}


def test_build_context_defaults_context_to_empty_dict():
    from linktools.ai.core.registry import AgentSpec

    kernel = _kernel()
    spec = AgentSpec(name="a", path=None, base_dir=None, enabled=True, model="standard")

    class _FakeSession:
        pass

    ctx = kernel.build_context(spec, _FakeSession(), builtin_tool_names=frozenset())

    assert ctx.context == {}
