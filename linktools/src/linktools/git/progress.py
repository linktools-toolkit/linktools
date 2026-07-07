#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Git progress parsing (spec §12.2 GitProgress)."""

import re

from linktools import utils

_PROGRESS_RE = re.compile(rb'^(.+?):\s+(?:\s*\d+%\s+\((\d+)/(\d+)\))?')


class GitProgressStream(object):
    """Writable stream that parses git progress output and updates rich bars."""

    def __init__(self, progress_ctx):
        self._progress = progress_ctx
        self._tasks = {}
        self._buf = b""

    def write(self, data):
        # type: (bytes) -> int
        self._buf += data
        while True:
            nl = self._buf.find(b'\n')
            cr = self._buf.find(b'\r')
            if nl == -1 and cr == -1:
                break
            idx = min(x for x in (nl, cr) if x != -1)
            line = self._buf[:idx].strip()
            self._buf = self._buf[idx + 1:]
            self._parse_line(line)
        return len(data)

    def flush(self):
        pass

    def _parse_line(self, line):
        if not line:
            return
        m = _PROGRESS_RE.match(line)
        if not m:
            return
        stage = m.group(1).decode("utf-8", errors="replace")
        cur = utils.int(m.group(2), default=None) if m.group(2) else None
        total = utils.int(m.group(3), default=None) if m.group(3) else None

        task_id = self._tasks.get(stage)
        if task_id is None:
            task_id = self._tasks[stage] = self._progress.add_task(
                stage, total=None, message=""
            )
        message = (
            "[progress.percentage]"
            "%s/%s" % (utils.coalesce(cur, "?"), utils.coalesce(total, "?"))
        )
        self._progress.update(task_id, message=message, completed=cur, total=total)
