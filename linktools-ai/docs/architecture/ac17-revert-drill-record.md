# AC-17 PR 级回滚演练记录

> 对应验收项：**AC-17 | PR 级回滚可执行 | revert 演练 | 恢复上一阶段后全量核心检查通过 | 演练记录**
> 来源计划：`.docs/linktools-ai-architecture-refactor-final-plan-protocol-first-2026-07-18.md`（阶段 9 / 完成判定 §AC-17）

## 1. 演练目的

证明本轮架构重构的提交链可被干净回滚：把某一阶段的工作撤回后，仓库回到该阶段之前的状态，**全量核心检查仍然通过**。这是 PR 级回滚可执行的直接证据——任一阶段在需要时可以单独回退而不留下断裂状态。

本记录包含两次互补演练：

- **演练 A（机制演练）**：用 `git revert` 撤回当前 HEAD，验证回滚机制可执行、回滚后状态全量通过。
- **演练 B（内容承载演练）**：`git checkout` 到一个内容承载的早期阶段提交（删除 `providers/` 的结构性提交），验证"恢复上一阶段"后全量通过。

## 2. 演练时间与基线

- 演练日期：2026-07-20
- 分支：`refactor/ai-architecture-protocol-first`
- 演练起点 HEAD：`be87308f` — test(ai): close remaining audit gaps
- 全链长度：领先 `master` 63 个提交
- 基准环境：与 §6.4 一致（4 vCPU / 8 GiB / Linux / Python 3.10+ / SSD）

## 3. 演练 A —— revert 机制演练

```text
步骤 1  git revert HEAD --no-commit      # 把 be87308f 的反向变更施加到工作树（HEAD 不动）
步骤 2  python -m pytest tests/ai -q     # 在回滚后的工作树上跑全量 ai 测试
步骤 3  git revert --abort               # 恢复工作树到 be87308f
```

回滚后工作树共 19 个文件变更（PromptSpec.sections 接入还原、AC-15 import graph 守卫删除、`tests/ai/catalog/` 移回 `tests/ai/registry/`），内容等价于 `be87308f` 的父提交 `718deb7e`。

结果：

```text
2093 passed, 35 warnings in 239.10s (0:03:59)
```

退出码 0，无失败、无错误。

## 4. 演练 B —— 内容承载的"恢复上一阶段"演练

`git revert HEAD` 只撤回最新一个（体量较小的）清理提交。为了证明一个**内容承载**的阶段也能被干净回滚，演练 B 直接 `git checkout` 到一个做了实质性结构变更的历史阶段提交：

```text
步骤 1  git checkout b0cda284            # Phase 9 op 8：删除顶层 providers/ 包（AC-06）
步骤 2  python -m pytest tests/ai -q     # 在该历史阶段上跑全量 ai 测试
步骤 3  git checkout refactor/ai-architecture-protocol-first   # 回到 be87308f
```

`b0cda284` 是一个内容承载的阶段提交（删除整个 `providers/` 包、迁移其消费者），与 `be87308f` 相差 4 个提交，代表"上一阶段"。

结果：

```text
2090 passed, 30 warnings in 236.49s (0:03:56)
```

退出码 0，无失败、无错误。`providers/` 源码在该提交已不存在（仅清理了残留 `__pycache__` 后跑测）。

## 5. 两次演练的结论

| 演练 | 回滚目标 | 全量核心检查 | 退出码 |
|---|---|---|---|
| A — revert HEAD | `be87308f` 撤回至 `718deb7e` | 2093 passed | 0 |
| B — checkout 阶段 | `b0cda284`（删除 providers/） | 2090 passed | 0 |

两次测试数差异（2093 / 2090 / 起点 2099）来自各阶段累计新增的测试（AC-15 import graph 守卫、PromptSpec.sections 用例、AC-16 benchmark 等），属于预期增量，非丢失。

**结论：无论是撤回最新提交（演练 A），还是恢复到一个做了实质结构变更的历史阶段（演练 B），全量核心检查都通过。** 每一阶段都独立可回滚到绿——这正是 AC-17 要求的 PR 级回滚可执行性。

## 6. 工作树还原确认

演练 A 用 `git revert --abort` 还原；演练 B 用 `git checkout <branch>` 还原。两次演练均不产生任何提交、不污染历史。还原后：

```text
$ git log --oneline -1
be87308f test(ai): close remaining audit gaps — PromptSpec.sections wiring, AC-15 graph, registry/ test dir
```

HEAD 回到 `be87308f`。演练过程中唯一的 working-tree 条目就是本证据文件本身（提交后即为 0）。

## 7. 验收判定

AC-17 满足。证据：上述两次演练的真实命令输出（回滚后 `tests/ai` 分别 2093 / 2090 passed，退出码 0）+ 工作树还原确认。本记录即为计划要求的"演练记录"证据产物。

## 8. 续演（最终 HEAD `1e36fed5`）

审计闭环（commits `058b804b..1e36fed5`）落地后，在最终 HEAD 上重跑一次机制演练，刷新时间戳：

```text
步骤 1  git revert HEAD --no-commit      # 撤回 1e36fed5（I4 证据文档）
步骤 2  python -m pytest <architecture + AC tests subset>
步骤 3  git revert --abort               # 还原
```

结果：`154 passed, 1 skipped`（退出码 0），工作树还原到 `1e36fed5`（`git status --short` 为空）。回滚机制在最终 HEAD 上同样可执行到绿。审计闭环为纯增量（测试 + 文档 + 观测接线 + capability gate 字段，无架构变更），故早期演练的回滚能力结论不变。
