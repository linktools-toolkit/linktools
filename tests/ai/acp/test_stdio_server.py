import inspect

from linktools.ai.acp.server import run_stdio, serve_stdio


def test_stdio_server_is_sdk_transport_entrypoint() -> None:
    assert inspect.iscoroutinefunction(serve_stdio)
    assert callable(run_stdio)
