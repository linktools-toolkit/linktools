#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import re
from typing import TYPE_CHECKING

from dulwich import porcelain
from dulwich.repo import Repo as DulwichRepo

from linktools import utils
from linktools.core import environ
from linktools.errors import GitError
from linktools.rich import create_progress

if TYPE_CHECKING:
    from typing import Any

_logger = environ.get_logger("git")

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

    def stash(self, *args: "Any"):
        if args and args[0] == "pop":
            porcelain.stash_pop(self._path)
        else:
            porcelain.stash_push(self._path)

    def reset(self, hard: bool = False, **kwargs: "Any"):
        porcelain.reset(self._path, mode="hard" if hard else "mixed")

    def checkout(self, branch: str):
        porcelain.switch(self._path, branch)


class GitHead:
    """A local git branch that can be checked out."""

    def __init__(self, path: str, name: str):
        self._path = path
        self.name = name

    def checkout(self):
        """Check out this branch in the working tree."""
        porcelain.switch(self._path, self.name)


class GitRepository:
    """Pure-Python git repository wrapper backed by dulwich.

    Provides the small subset of git operations needed to clone, track,
    and keep a shallow (depth=1) checkout in sync with its remote.
    """

    def __init__(self, path: "Any"):
        self._path = str(path)
        self._repo = DulwichRepo(self._path)  # raises NotGitRepository if invalid
        self.git = _GitProxy(self._path)

    @property
    def heads(self) -> "list[str]":
        """Return the names of all local branches."""
        refs = self._repo.refs.as_dict(b"refs/heads/")
        return [name.decode() for name in refs]

    def status(self):
        """Return the working tree status (staged/unstaged/untracked files)."""
        return porcelain.status(self._path)

    def is_dirty(self) -> bool:
        """Return whether the working tree has staged or unstaged changes."""
        status = self.status()
        return bool(any(status.staged.values()) or status.unstaged)

    def add(self, *paths: str):
        """Stage files for the next commit.

        Args:
            paths: Paths to stage. Stages all changes in the working tree
                if omitted.
        """
        porcelain.add(self._path, list(paths) or None)

    def commit(self, message: str, *, author: "str | None" = None, committer: "str | None" = None, all: bool = False) -> str:
        """Create a commit from the currently staged changes.

        Args:
            message: Commit message.
            author: Commit author, e.g. "Name <email>". Defaults to the
                repository's configured user.
            committer: Commit committer. Defaults to `author`.
            all: If True, automatically stage all modified tracked files
                before committing (like `git commit -a`).

        Returns:
            str: The hex sha of the new commit.
        """
        sha = porcelain.commit(
            self._path,
            message=message,
            author=author.encode() if author else None,
            committer=committer.encode() if committer else None,
            all=all,
        )
        return sha.decode()

    def push(self, remote_location: "str | None" = None, branch: "str | None" = None, *, force: bool = False):
        """Push a branch to its remote.

        Args:
            remote_location: Remote name or URL. Defaults to the branch's
                configured remote, falling back to `origin`.
            branch: Local branch to push. Defaults to the current branch.
            force: If True, allow non-fast-forward updates.
        """
        refspecs = branch and self._branch_ref(branch).decode()
        porcelain.push(self._path, remote_location, refspecs, force=force)

    def create_head(self, branch: str) -> "GitHead":
        """Create (and fetch if necessary) a local branch tracking the remote branch.

        Args:
            branch: Name of the branch to create.

        Returns:
            GitHead: The newly created local branch.

        Raises:
            GitError: If the branch cannot be found on the remote.
        """
        branch_ref = self._branch_ref(branch)
        target = self._remote_branch_target(branch)
        if target is None:
            result = porcelain.fetch(self._path, depth=1, force=True, quiet=True)
            target = result.refs.get(branch_ref)
        if target is None:
            raise GitError(f"Remote branch `{branch}` not found.")
        porcelain.branch_create(self._path, branch, target)
        return GitHead(self._path, branch)

    def pull(self, reset: bool = False):
        """Pull the current branch from its remote, reporting progress.

        Args:
            reset: If True, hard-reset the branch to match the remote instead
                of attempting a fast-forward pull. Required when the local
                branch has diverged from a shallow (depth=1) remote, since
                dulwich cannot merge or rebase in that case.

        Raises:
            GitError: If the branch has diverged and `reset` is False.
        """
        with create_progress("message") as progress:
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
                raise GitError(
                    "Local branch has diverged from the remote and cannot be "
                    "fast-forwarded. Re-run with force reset enabled to reset it to the remote."
                )

    def _branch_ref(self, branch: str) -> bytes:
        return b"refs/heads/" + branch.encode()

    def _current_branch_ref(self) -> bytes:
        head_refs, _ = self._repo.refs.follow(b"HEAD")
        branch_ref = head_refs[-1]
        if not branch_ref.startswith(b"refs/heads/"):
            raise GitError("Repository HEAD is detached; unable to resolve branch to update.")
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
        target = result.refs.get(branch_ref)
        if target is None:
            raise GitError(f"Remote branch `{branch_ref.decode()}` not found.")
        porcelain.reset(self._path, "hard", target)

    @classmethod
    def clone(cls, url: str, repo_path: "str | None" = None, branch: "str | None" = None):
        """Shallow-clone a repository, reporting progress.

        Args:
            url: Repository URL to clone.
            repo_path: Destination path for the clone.
            branch: Branch to check out after cloning, if not the default.
        """
        kwargs = {}
        if branch:
            kwargs["branch"] = branch
        with create_progress("message") as progress:
            porcelain.clone(
                url, 
                repo_path, 
                depth=1, 
                errstream=_ProgressStream(progress), 
                **kwargs
            )
