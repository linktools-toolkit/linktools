#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


path = Path("linktools-ai/src/linktools/ai/runtime/state/_repositories.py")
text = path.read_text(encoding="utf-8")
old = '''        candidate.owner_principal_id,
        candidate.agent_id,
        candidate.history_id,
    ) != (
        current.session_id,
        current.tenant_id,
        current.owner_principal_id,
        current.agent_id,
        current.history_id,
    ):
'''
new = '''        candidate.owner_principal_id,
        candidate.agent_id,
        candidate.history_id,
        candidate.created_at,
    ) != (
        current.session_id,
        current.tenant_id,
        current.owner_principal_id,
        current.agent_id,
        current.history_id,
        current.created_at,
    ):
'''
if text.count(old) != 1:
    raise RuntimeError("session identity tuple did not match exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
