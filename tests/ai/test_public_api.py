#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import linktools.ai as ai


def test_public_api_exports_agent_spec():
    from linktools.ai.agent_runtime.spec import AgentSpec as _Real
    assert ai.AgentSpec is _Real


def test_public_api_exports_runtime():
    from linktools.ai.runtime import Runtime as _Real
    assert ai.Runtime is _Real


def test_public_api_exports_file_storage():
    from linktools.ai.storage.facade import FileStorage as _Real
    assert ai.FileStorage is _Real


def test_public_api_exports_sqlalchemy_storage():
    from linktools.ai.storage.facade import SqlAlchemyStorage as _Real
    assert ai.SqlAlchemyStorage is _Real


def test_public_api_exports_storage():
    from linktools.ai.storage.facade import Storage as _Real
    assert ai.Storage is _Real


def test_public_api_does_not_re_export_internals():
    for internal in ("AgentCompiler", "AgentRunner", "CompiledAgent", "Middleware",
                     "MiddlewarePipeline", "PolicyEngine", "ToolExecutor", "RunStore",
                     "SessionStore", "EventStore", "ResourceStore", "ModelRouter"):
        assert not hasattr(ai, internal), f"linktools.ai should not export {internal}"
