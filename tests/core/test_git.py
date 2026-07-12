#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import shutil
import tempfile
import unittest

from dulwich import porcelain
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo as DulwichRepo

from linktools.errors import GitError
from linktools.git import GitRepository, GitSyncPolicy
from linktools.core import environ

_AUTHOR = b"Test User <test@example.com>"


class TestGit(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="linktools-git-test-")
        self.remote_path = os.path.join(self._tmp_dir, "remote")
        self.clone_path = os.path.join(self._tmp_dir, "clone")

        os.makedirs(self.remote_path)
        porcelain.init(self.remote_path)
        self._commit(self.remote_path, "a.txt", "hello", "first")

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @staticmethod
    def _commit(path: str, filename: str, content: str, message: str):
        file_path = os.path.join(path, filename)
        with open(file_path, "w") as f:
            f.write(content)
        porcelain.add(path, [file_path])
        porcelain.commit(path, message=message.encode(), author=_AUTHOR, committer=_AUTHOR)

    @staticmethod
    def _read(path: str, filename: str) -> str:
        with open(os.path.join(path, filename)) as f:
            return f.read()

    def test_clone(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        self.assertTrue(os.path.isdir(os.path.join(self.clone_path, ".git")))
        self.assertEqual(self._read(self.clone_path, "a.txt"), "hello")

    def test_clone_branch(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path, branch="feature")
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertIn("feature", repo.heads)

    def test_heads(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        # Single-branch shallow clone only fetches the default branch.
        self.assertEqual(repo.heads, ["master"])

    def test_is_dirty(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertFalse(repo.is_dirty())

        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("changed")
        self.assertTrue(repo.is_dirty())

    def test_head_sha_matches_remote_after_clone(self):
        with DulwichRepo(self.remote_path) as remote_repo:
            expected = remote_repo.head().decode()

        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        sha = repo.head_sha()
        self.assertEqual(sha, expected)
        self.assertEqual(len(sha), 40)

    def test_head_sha_changes_after_commit(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        before = repo.head_sha()

        with open(os.path.join(self.clone_path, "b.txt"), "w") as f:
            f.write("new file")
        repo.add("b.txt")
        after = repo.commit("add b.txt", author="Test <test@example.com>")

        self.assertNotEqual(before, after)
        self.assertEqual(repo.head_sha(), after)

    def test_current_branch_on_default_branch(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertEqual(repo.current_branch(), "master")

    def test_current_branch_after_checkout_or_create(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        repo.checkout_or_create("feature")

        self.assertEqual(repo.current_branch(), "feature")

    def test_current_branch_is_none_when_detached(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        sha = repo.head_sha()
        porcelain.update_head(self.clone_path, sha, detached=True)

        self.assertIsNone(repo.current_branch())

    def test_status(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("changed")
        with open(os.path.join(self.clone_path, "b.txt"), "w") as f:
            f.write("new file")

        status = repo.status()
        self.assertIn(b"a.txt", status.unstaged)
        self.assertIn(b"b.txt", status.untracked)

        repo.add("b.txt")
        self.assertIn(b"b.txt", repo.status().staged["add"])

    def test_commit(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        with open(os.path.join(self.clone_path, "b.txt"), "w") as f:
            f.write("new file")
        repo.add("b.txt")
        sha = repo.commit("add b.txt", author="Test <test@example.com>")

        self.assertEqual(len(sha), 40)
        self.assertFalse(repo.is_dirty())

    def test_commit_all(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("changed")
        repo.commit("update a.txt", author="Test <test@example.com>", all=True)

        self.assertFalse(repo.is_dirty())

    def test_push(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        with open(os.path.join(self.clone_path, "b.txt"), "w") as f:
            f.write("new file")
        repo.add("b.txt")
        sha = repo.commit("add b.txt", author="Test <test@example.com>")

        repo.push()

        with DulwichRepo(self.remote_path) as remote_repo:
            remote_heads = remote_repo.refs.as_dict(b"refs/heads/")
        self.assertEqual(remote_heads[b"master"].decode(), sha)

    def test_checkout_or_create_switches_existing_branch(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        porcelain.branch_create(self.clone_path, "local-branch")

        repo.checkout_or_create("local-branch")
        self.assertEqual(porcelain.active_branch(self.clone_path), b"local-branch")

        repo.checkout_or_create("master")
        self.assertEqual(porcelain.active_branch(self.clone_path), b"master")

    def test_checkout_or_create_creates_missing_branch_from_remote(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertNotIn("feature", repo.heads)

        repo.checkout_or_create("feature")

        self.assertIn("feature", repo.heads)
        self.assertEqual(porcelain.active_branch(self.clone_path), b"feature")

    def test_sync_fast_forward_only(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        self._commit(self.remote_path, "b.txt", "world", "second")
        repo.sync(policy=GitSyncPolicy.FAST_FORWARD_ONLY)

        self.assertEqual(self._read(self.clone_path, "b.txt"), "world")

    def test_sync_stash_and_restore_policy(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        self._commit(self.remote_path, "b.txt", "world", "second")
        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("dirty change")

        repo.sync(policy=GitSyncPolicy.STASH_AND_RESTORE)

        # Fast-forwarded to the remote's new commit...
        self.assertEqual(self._read(self.clone_path, "b.txt"), "world")
        # ...and the dirty local change was restored afterwards.
        self.assertEqual(self._read(self.clone_path, "a.txt"), "dirty change")

    def test_sync_fast_forward_diverged_raises(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        self._commit(self.remote_path, "b.txt", "remote change", "remote second")
        self._commit(self.clone_path, "c.txt", "local change", "local second")

        with self.assertRaises(GitError):
            repo.sync(policy=GitSyncPolicy.FAST_FORWARD_ONLY)

    def test_create_head_from_remote_branch(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertNotIn("feature", repo.heads)

        head = repo.create_head("feature")
        head.checkout()

        self.assertIn("feature", repo.heads)

    def test_create_head_missing_branch_raises(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        with self.assertRaises(GitError):
            repo.create_head("does-not-exist")

    def test_invalid_repository_raises(self):
        not_a_repo = os.path.join(self._tmp_dir, "not-a-repo")
        os.makedirs(not_a_repo)

        with self.assertRaises(NotGitRepository):
            GitRepository(environ, not_a_repo)

    def test_clone_target_exists_raises(self):
        # Atomic clone (§12.2): refuse to clobber an existing target rather
        # than leaving a half-merged repository.
        os.makedirs(self.clone_path)
        with self.assertRaises(GitError):
            GitRepository.clone(environ, self.remote_path, self.clone_path)

    def test_clone_failure_leaves_no_partial(self):
        # If the clone fails mid-way, no staging dir or partial repo survives
        # at the target (spec §27.2: interrupted clone never produces a valid
        # repository).
        from unittest import mock

        with mock.patch("linktools.git.repository.porcelain.clone",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                GitRepository.clone(environ, self.remote_path, self.clone_path)
        self.assertFalse(os.path.exists(self.clone_path))
        leftovers = [p for p in os.listdir(self._tmp_dir)
                     if p.startswith("clone.staging-")]
        self.assertEqual(leftovers, [])

    def test_clone_protocol_error_is_wrapped_as_git_error(self):
        # A transport/protocol failure (bad URL, auth rejected, server
        # error, ...) must surface as a plain GitError -- callers must
        # never need to import dulwich just to catch its own exception type.
        from unittest import mock
        from dulwich.errors import GitProtocolError

        with mock.patch("linktools.git.repository.porcelain.clone",
                        side_effect=GitProtocolError("bad refs")):
            with self.assertRaises(GitError):
                GitRepository.clone(environ, self.remote_path, self.clone_path)
        self.assertFalse(os.path.exists(self.clone_path))

    def test_sync_reset_policy(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self._commit(self.remote_path, "b.txt", "remote change", "remote second")
        self._commit(self.clone_path, "c.txt", "local change", "local second")
        repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE)
        self.assertEqual(self._read(self.clone_path, "b.txt"), "remote change")
        self.assertFalse(os.path.exists(os.path.join(self.clone_path, "c.txt")))


if __name__ == '__main__':
    unittest.main()
