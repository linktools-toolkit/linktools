from pathlib import Path
from types import SimpleNamespace

from linktools.cli.env import get_commands


class Java:
    version = "17.0.8"

    def __init__(self, version=None):
        if version:
            self.version = version

    def copy(self, **overrides):
        return Java(overrides["version"])

    def get_variable(self, name):
        assert name == "home_path"
        return "/opt/java/%s" % self.version

    def make_cmdargs(self):
        return ["linktools", "tool", "java"]


class Environ:
    name = "test"
    version = "1"
    system = "linux"

    def __init__(self, root):
        self.root = Path(root)
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None,
                                      warning=lambda *args, **kwargs: None)

    def get_data_path(self, *parts, **kwargs):
        return self.root.joinpath(*parts)

    def get_tool(self, name):
        assert name == "java"
        return Java()


def test_java_command_uses_public_tool_properties(capsys, tmp_path):
    java = next(item for item in get_commands(Environ(tmp_path)) if item.name == "java")
    java.run(SimpleNamespace(shell="bash", version="17.0.11"))
    output = capsys.readouterr().out
    assert "JAVA_VERSION='17.0.11'" in output
    assert "JAVA_HOME='/opt/java/17.0.11'" in output
