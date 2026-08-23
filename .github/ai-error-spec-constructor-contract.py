from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


path = Path("linktools-ai/src/linktools/ai/spec/_contract.py")
text = path.read_text()

text = replace_once(
    text,
    '''        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
            for value in values
        ):
            raise ValueError("usage limits must be positive integers or None")
''',
    '''        if any(
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in values
        ):
            raise TypeError("usage limits must contain integers or None")
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("usage limits must be positive integers")
''',
    label="AgentUsageLimits",
)

text = replace_once(
    text,
    '''        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("agent id must be a non-empty string")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("agent revision must be a positive integer")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent model must be a non-empty string")
''',
    '''        if not isinstance(self.id, str):
            raise TypeError("agent id must be a string")
        if not self.id.strip():
            raise ValueError("agent id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("agent revision must be an integer")
        if self.revision < 1:
            raise ValueError("agent revision must be positive")
        if not isinstance(self.model, str):
            raise TypeError("agent model must be a string")
        if not self.model.strip():
            raise ValueError("agent model must be non-empty")
''',
    label="AgentSpec identity",
)

text = replace_once(
    text,
    '''    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
            or not isinstance(self.content, str)
        ):
            raise ValueError("skill spec is incomplete")
''',
    '''    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("skill id must be a string")
        if not self.id.strip():
            raise ValueError("skill id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("skill revision must be an integer")
        if self.revision < 1:
            raise ValueError("skill revision must be positive")
        if not isinstance(self.content, str):
            raise TypeError("skill content must be a string")
''',
    label="SkillSpec",
)

text = replace_once(
    text,
    '''    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
            or not isinstance(self.command, str)
            or not self.command.strip()
            or isinstance(self.args, (str, bytes, bytearray))
            or not isinstance(self.args, Sequence)
        ):
            raise ValueError("MCP server spec is incomplete")
        args = tuple(self.args)
        if any(not isinstance(item, str) for item in args):
            raise ValueError("MCP server args must be strings")
        object.__setattr__(self, "args", args)
''',
    '''    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("MCP server id must be a string")
        if not self.id.strip():
            raise ValueError("MCP server id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("MCP server revision must be an integer")
        if self.revision < 1:
            raise ValueError("MCP server revision must be positive")
        if not isinstance(self.command, str):
            raise TypeError("MCP server command must be a string")
        if not self.command.strip():
            raise ValueError("MCP server command must be non-empty")
        if isinstance(self.args, (str, bytes, bytearray)) or not isinstance(self.args, Sequence):
            raise TypeError("MCP server args must be a string sequence")
        args = tuple(self.args)
        if any(not isinstance(item, str) for item in args):
            raise TypeError("MCP server args must be strings")
        object.__setattr__(self, "args", args)
''',
    label="MCPServerSpec",
)

path.write_text(text)


test_path = Path("tests/ai/test_spec_capability_refactor.py")
test_text = test_path.read_text()
test_text = replace_once(
    test_text,
    '''    AgentSpec,
    AgentSpecCodec,
''',
    '''    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
''',
    label="AgentUsageLimits import",
)
test_text = replace_once(
    test_text,
    '''def test_spec_constructors_reject_mismatched_runtime_types() -> None:
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "default", instructions="not-an-array")
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "default", usage_limits=object())
    with pytest.raises(ValueError):
        SkillSpec("skill", True, "content")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 1, "echo", args="not-an-array")


''',
    '''def test_spec_constructors_reject_mismatched_runtime_types() -> None:
    with pytest.raises(TypeError):
        AgentUsageLimits(model_requests=True)
    with pytest.raises(TypeError):
        AgentSpec(1, 1, "default")
    with pytest.raises(TypeError):
        AgentSpec("agent", True, "default")
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, 1)
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", instructions="not-an-array")
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", instructions=(1,))
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", metadata=[])
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", usage_limits=object())
    with pytest.raises(TypeError):
        SkillSpec(1, 1, "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", True, "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", 1, 1)
    with pytest.raises(TypeError):
        MCPServerSpec(1, 1, "echo")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", True, "echo")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, 1)
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, "echo", args="not-an-array")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, "echo", args=(1,))


def test_spec_constructors_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        AgentUsageLimits()
    with pytest.raises(ValueError):
        AgentUsageLimits(model_requests=0)
    with pytest.raises(ValueError):
        AgentSpec("", 1, "default")
    with pytest.raises(ValueError):
        AgentSpec("agent", 0, "default")
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "")
    with pytest.raises(ValueError):
        SkillSpec("", 1, "content")
    with pytest.raises(ValueError):
        SkillSpec("skill", 0, "content")
    with pytest.raises(ValueError):
        MCPServerSpec("", 1, "echo")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 0, "echo")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 1, "")


''',
    label="spec constructor tests",
)
test_path.write_text(test_text)
