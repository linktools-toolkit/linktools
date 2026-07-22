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
        # Atomic clone: refuse to clobber an existing target rather
        # than leaving a half-merged repository.
        os.makedirs(self.clone_path)
        with self.assertRaises(GitError):
            GitRepository.clone(environ, self.remote_path, self.clone_path)

    def test_clone_failure_leaves_no_partial(self):
        # If the clone fails mid-way, no staging dir or partial repo survives
        # at the target (: interrupted clone never produces a valid
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

    # -- P2-01: untracked files count as dirty -------------------------------

    def test_is_dirty_staged(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("staged change")
        porcelain.add(self.clone_path, [os.path.join(self.clone_path, "a.txt")])
        self.assertTrue(repo.is_dirty())

    def test_is_dirty_unstaged(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("unstaged change")
        self.assertTrue(repo.is_dirty())

    def test_is_dirty_untracked_only(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        # A brand new, never-added file -- neither staged nor unstaged
        # (dulwich's unstaged only covers already-tracked files).
        with open(os.path.join(self.clone_path, "new_file.txt"), "w") as f:
            f.write("untracked")
        self.assertTrue(repo.is_dirty())

    def test_is_dirty_clean_repo_is_false(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self.assertFalse(repo.is_dirty())

    # -- P2-02: every write goes through the same repo write lock -----------

    def test_checkout_via_head_is_serialized_by_write_lock(self):
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        head = repo.create_head("feature")

        calls = []
        real_write_lock = repo._write_lock

        def spy_write_lock():
            calls.append(1)
            return real_write_lock()

        repo._write_lock = spy_write_lock
        head.checkout()
        self.assertEqual(calls, [1])

    def test_checkout_or_create_acquires_the_lock_exactly_once(self):
        """Verifies WP2-02's fix directly: checkout_or_create's create
        branch must not nest two process_lock() acquisitions (which would
        deadlock a real file-based lock) -- it must acquire the lock
        exactly once for the whole operation."""
        porcelain.branch_create(self.remote_path, "feature")
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)

        calls = []
        real_write_lock = repo._write_lock

        def spy_write_lock():
            calls.append(1)
            return real_write_lock()

        repo._write_lock = spy_write_lock
        repo.checkout_or_create("feature")
        self.assertEqual(calls, [1])

    def test_add_is_serialized_by_write_lock(self):
        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        with open(os.path.join(self.clone_path, "new_file.txt"), "w") as f:
            f.write("content")

        calls = []
        real_write_lock = repo._write_lock

        def spy_write_lock():
            calls.append(1)
            return real_write_lock()

        repo._write_lock = spy_write_lock
        repo.add(os.path.join(self.clone_path, "new_file.txt"))
        self.assertEqual(calls, [1])

    # -- P2-03: stash-restore failure must not mask the original error ------

    def test_sync_failure_and_stash_restore_failure_both_surface(self):
        from unittest import mock
        from linktools.errors import GitStashRestoreError

        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self._commit(self.remote_path, "b.txt", "remote change", "remote second")
        self._commit(self.clone_path, "c.txt", "local change", "local second")
        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("dirty change")

        with mock.patch.object(repo, "_stash_pop", side_effect=RuntimeError("pop boom")):
            with self.assertRaises(GitStashRestoreError) as ctx:
                repo.sync(policy=GitSyncPolicy.STASH_AND_RESTORE)

        message = str(ctx.exception)
        self.assertIn("pop boom", message)
        # The original sync failure (diverged branch) must still be visible,
        # not silently replaced by the restore failure.
        self.assertTrue("diverged" in message.lower() or "GitDivergedError" in message)
        self.assertIsInstance(ctx.exception.__cause__, Exception)

    def test_stash_restore_failure_alone_raises_after_successful_sync(self):
        from unittest import mock
        from linktools.errors import GitStashRestoreError

        GitRepository.clone(environ, self.remote_path, self.clone_path)
        repo = GitRepository(environ, self.clone_path)
        self.addCleanup(repo.close)
        self._commit(self.remote_path, "b.txt", "remote change", "remote second")
        with open(os.path.join(self.clone_path, "a.txt"), "w") as f:
            f.write("dirty change")

        with mock.patch.object(repo, "_stash_pop", side_effect=RuntimeError("pop boom")):
            with self.assertRaises(GitStashRestoreError) as ctx:
                repo.sync(policy=GitSyncPolicy.STASH_AND_RESTORE)

        self.assertIn("pop boom", str(ctx.exception))
        # The fast-forward itself succeeded.
        self.assertEqual(self._read(self.clone_path, "b.txt"), "remote change")

    # -- clone rejects a dangling symlink at the target ----------------------

    def test_clone_target_dangling_symlink_raises(self):
        missing = os.path.join(self._tmp_dir, "does-not-exist")
        os.symlink(missing, self.clone_path)
        self.assertFalse(os.path.exists(self.clone_path))  # dangling
        with self.assertRaises(GitError):
            GitRepository.clone(environ, self.remote_path, self.clone_path)

    # -- open_if_valid broadened beyond NotGitRepository ---------------------

    def test_open_if_valid_returns_none_for_non_repository(self):
        not_a_repo = os.path.join(self._tmp_dir, "not-a-repo-2")
        os.makedirs(not_a_repo)
        self.assertIsNone(GitRepository.open_if_valid(environ, not_a_repo))

    def test_open_if_valid_returns_none_on_any_open_failure(self):
        from unittest import mock

        GitRepository.clone(environ, self.remote_path, self.clone_path)
        with mock.patch("linktools.git.repository.DulwichRepo",
                        side_effect=RuntimeError("corrupt object store")):
            self.assertIsNone(GitRepository.open_if_valid(environ, self.clone_path))


if __name__ == '__main__':
    unittest.main()
