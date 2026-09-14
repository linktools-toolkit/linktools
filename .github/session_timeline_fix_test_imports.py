from pathlib import Path

for name in ("test_session_timeline.py", "test_session_timeline_fork.py"):
    path = Path("tests/ai") / name
    text = path.read_text(encoding="utf-8")
    old = "from linktools.ai.agent import AgentBindingSnapshot, AgentSpec, bind_output\n"
    new = (
        "from linktools.ai.agent import AgentBindingSnapshot\n"
        "from linktools.ai.agent._output import bind_output\n"
        "from linktools.ai.spec import AgentSpec\n"
    )
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: invalid import marker")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
