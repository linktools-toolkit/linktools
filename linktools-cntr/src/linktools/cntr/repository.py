#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : repository.py
@time    : 2024/3/24
@site    : https://github.com/ice-black-tea
@software: PyCharm

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
import re

from dulwich import porcelain
from dulwich.repo import Repo as DulwichRepo

from linktools import utils
from linktools.rich import create_simple_progress
from .container import ContainerError

_PROGRESS_RE = re.compile(rb'^(.+?):\s+(?:\s*\d+%\s+\((\d+)/(\d+)\))?')


class _ProgressStream:
    """Writable stream that parses git progress output and updates rich progress bars."""

    def __init__(self, progress_ctx):
        self._progress = progress_ctx
        self._tasks = {}
        self._buf = b""

    def write(self, data: bytes) -> int:
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

    def _parse_line(self, line: bytes):
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
            f"[progress.percentage]"
            f"{utils.coalesce(cur, '?')}/"
            f"{utils.coalesce(total, '?')}"
        )
        self._progress.update(task_id, message=message, completed=cur, total=total)


class _GitProxy:
    """Proxy that exposes common git operations on a repo path via dulwich porcelain."""

    def __init__(self, path: str):
        self._path = path

    def stash(self, *args):
        if args and args[0] == "pop":
            porcelain.stash_pop(self._path)
        else:
            porcelain.stash_push(self._path)

    def reset(self, hard: bool = False, **kwargs):
        porcelain.reset(self._path, mode="hard" if hard else "mixed")

    def checkout(self, branch: str):
        porcelain.switch(self._path, branch)


class _Head:

    def __init__(self, path: str, name: str):
        self._path = path
        self.name = name

    def checkout(self):
        porcelain.switch(self._path, self.name)


class Repository:

    def __init__(self, path):
        self._path = str(path)
        self._repo = DulwichRepo(self._path)  # raises NotGitRepository if invalid
        self.git = _GitProxy(self._path)

    @property
    def heads(self):
        refs = self._repo.refs.as_dict(b"refs/heads/")
        return [name.decode() for name in refs]

    def is_dirty(self) -> bool:
        status = porcelain.status(self._path)
        staged = status.staged
        unstaged = status.unstaged
        return bool(any(staged.values()) or unstaged)

    def create_head(self, branch: str) -> _Head:
        branch_ref = self._branch_ref(branch)
        target = self._remote_branch_target(branch)
        if target is None:
            result = porcelain.fetch(self._path, depth=1, force=True, quiet=True)
            target = result.refs.get(branch_ref)
        if target is None:
            raise ContainerError(f"Remote branch `{branch}` not found.")
        porcelain.branch_create(self._path, branch, target)
        return _Head(self._path, branch)

    def update_with_progress(self, reset: bool = False):
        with create_simple_progress("message") as progress:
            if reset:
                self._force_update(progress)
                return
            branch_ref = self._current_branch_ref()
            try:
                porcelain.pull(
                    self._path,
                    refspecs=branch_ref,
                    errstream=_ProgressStream(progress),
                )
            except porcelain.DivergedBranches:
                raise ContainerError(
                    "Local branch has diverged from the remote and cannot be "
                    "fast-forwarded. Re-run with `--force` to reset it to the remote."
                )

    def _branch_ref(self, branch: str) -> bytes:
        return b"refs/heads/" + branch.encode()

    def _current_branch_ref(self) -> bytes:
        head_refs, _ = self._repo.refs.follow(b"HEAD")
        branch_ref = head_refs[-1]
        if not branch_ref.startswith(b"refs/heads/"):
            raise ContainerError("Repository HEAD is detached; unable to resolve branch to update.")
        return branch_ref

    def _remote_branch_target(self, branch: str):
        remote_ref = b"refs/remotes/origin/" + branch.encode()
        return self._repo.refs.as_dict().get(remote_ref)

    def _force_update(self, progress):
        # These repos are shallow (depth=1) clones, so dulwich cannot merge or
        # rebase a diverged branch. A forced update instead fetches the remote
        # objects and hard-resets the local branch to match the remote.
        branch_ref = self._current_branch_ref()  # e.g. b"refs/heads/master"
        result = porcelain.fetch(
            self._path,
            errstream=_ProgressStream(progress),
            depth=1,
            force=True,
        )
        target = result.refs.get(branch_ref) or result.refs.get(b"HEAD")
        if target is None:
            raise ContainerError("Unable to resolve the remote branch to reset to.")
        porcelain.reset(self._path, "hard", target)

    @classmethod
    def clone_with_progress(cls, url: str, repo_path: str = None, branch: str = None):
        kwargs = {}
        if branch:
            kwargs["branch"] = branch
        with create_simple_progress("message") as progress:
            porcelain.clone(url, repo_path, depth=1, errstream=_ProgressStream(progress), **kwargs)
