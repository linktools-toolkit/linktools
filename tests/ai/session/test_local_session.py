#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

from linktools.ai.session.local import local_session
from linktools.ai.session.types import FileSession


def test_local_session_returns_file_session_with_given_id():
    session = local_session("test-local-session-xyz")
    try:
        assert isinstance(session, FileSession)
        assert session.session_id == "test-local-session-xyz"
        assert session.root.name == "test-local-session-xyz"
        assert "sessions" in session.root.parts
        assert "ai" in session.root.parts
    finally:
        shutil.rmtree(session.root, ignore_errors=True)
