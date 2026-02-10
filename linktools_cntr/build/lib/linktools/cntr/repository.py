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
from git import Repo, RemoteProgress, GitCommandError
from git.util import CallableRemoteProgress

from linktools import utils
from linktools.rich import create_simple_progress


class Repository(Repo):

    def update_with_progress(self):

        try:
            with create_simple_progress("message") as progress:

                tasks = {}
                def update_progress(op_code, cur_count, max_count, git_message):
                    op_name = self._get_op_name(op_code)
                    if not op_name:
                        return

                    task_id = tasks.get(op_name)
                    if task_id is None:
                        task_id = tasks[op_name] = progress.add_task(
                            op_name,
                            total=None,
                            message=""
                        )

                    max_count = utils.int(max_count, default=None)
                    cur_count = utils.int(cur_count, default=None)

                    message = f"[progress.percentage]" \
                              f"{utils.coalesce(cur_count, '?')}/" \
                              f"{utils.coalesce(max_count, '?')}"
                    if git_message:
                        message += f" [progress.data.speed]{git_message}"

                    progress.update(
                        task_id,
                        message=message,
                        completed=cur_count,
                        total=max_count
                    )

                self.remote().pull(
                    progress=CallableRemoteProgress(update_progress),
                    allow_unsafe_protocols=True,
                    rebase=True,
                )

        except GitCommandError as e:
            if e.stderr and "unable to auto-detect email address" in e.stderr:
                pass
            raise e

    @classmethod
    def clone_with_progress(cls, url: str, repo_path: str = None, branch: str = None):

        try:
            with create_simple_progress("message") as progress:

                tasks = {}
                options = dict(depth="1")
                if branch:
                    options["branch"] = branch

                def update_progress(op_code, cur_count, max_count, git_message):
                    op_name = cls._get_op_name(op_code)
                    if not op_name:
                        return

                    task_id = tasks.get(op_name)
                    if task_id is None:
                        task_id = tasks[op_name] = progress.add_task(
                            op_name,
                            total=None,
                            message=""
                        )

                    max_count = utils.int(max_count, default=None)
                    cur_count = utils.int(cur_count, default=None)

                    message = f"[progress.percentage]" \
                              f"{utils.coalesce(cur_count, '?')}/" \
                              f"{utils.coalesce(max_count, '?')}"
                    if git_message:
                        message += f" [progress.data.speed]{git_message}"

                    progress.update(
                        task_id,
                        message=message,
                        completed=cur_count,
                        total=max_count
                    )

                cls.clone_from(
                    url,
                    repo_path,
                    progress=update_progress,
                    allow_unsafe_protocols=True,
                    **options
                )

        except GitCommandError as e:
            if e.stderr and "unable to auto-detect email address" in e.stderr:
                pass
            raise e

    @classmethod
    def _get_op_name(cls, op_code: int) -> str:
        op_name = ""
        if op_code & RemoteProgress.COUNTING:
            op_name = "Counting objects"
        elif op_code & RemoteProgress.COMPRESSING:
            op_name = "Compressing objects"
        elif op_code & RemoteProgress.WRITING:
            op_name = "Writing objects"
        elif op_code & RemoteProgress.RECEIVING:
            op_name = "Receiving objects"
        elif op_code & RemoteProgress.RESOLVING:
            op_name = "Resolving deltas"
        elif op_code & RemoteProgress.FINDING_SOURCES:
            op_name = "Finding sources"
        elif op_code & RemoteProgress.CHECKING_OUT:
            op_name = "Checking out files"
        return op_name
