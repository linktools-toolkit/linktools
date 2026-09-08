# linktools-ai Harness 回迁后整体 Review 修复规范

## 1. 目标

本规范用于修复 `refactor/ai-harness-capabilities` 分支在 Harness 回迁后的整体 Review 问题。

目标只有三个：

1. 消除已经确认的功能降级；
2. 删除已经失效、重复或错误归属的抽象和防御；
3. 保持 Harness / LinkTools 的职责边界，不重新实现 Harness 已提供的通用 capability。

本轮不得以“顺手优化”为理由扩大范围，不重新设计 Session、Execution、ToolOperation、Storage、Skill、Subagent、Task 等已经稳定的领域。

## 2. 不变量

### 2.1 Harness ownership

以下通用 capability 行为继续由 Harness 负责：

- `StepPersistence`
- `Planning`
- `Memory`
- Compaction：`DeduplicateFileReads`、`ClearToolResults`、`SummarizingCompaction`、`TieredCompaction`

不得重新复制上述 capability 的工具 schema、prompt、CAS retry、planning behavior、summarization algorithm。

### 2.2 LinkTools ownership

以下 durable/domain/isolation 事实继续由 LinkTools 负责：

- `RuntimeState` 与 FS / SQLite / MySQL / PostgreSQL persistence；
- Session / Execution / Recovery；
- `ToolOperation`、fencing、idempotency、effect-unknown；
- Memory 的 Runtime-backed durable store；
- Planning 的 Runtime-backed durable store；
- raw transcript 与 model-context projection；
- Metrics / model request journal；
- `Sandbox` / `SandboxSession` 隔离边界；
- Bubblewrap namespace、guardian/worker、resource projection；
- scoped repository instruction pre-tool gate。

### 2.3 并发与持久化

- 文件协调继续使用 `filelock`。
- 数据库禁止悲观锁。
- caller cancellation 不直接决定 durable truth。
- 一个 semantic fact 只有一个 durable owner。
- 不持久化 Harness DTO 作为 LinkTools durable authority。

## 3. 必须修复的问题

### F1 — Workspace 模型可纠正错误必须恢复为 ModelRetry

`SandboxSession` 保持 LinkTools `AIError` 低层合同；Pydantic-facing workspace adapter 负责把明确可由模型修正的 workspace 输入/访问错误转换为 `ModelRetry`。

模型可纠正错误至少包括：

- `REQUEST_FIELD_INVALID`
- `STORAGE_NOT_FOUND`
- `STORAGE_CONFLICT`
- `AUTHORIZATION_DENIED`

以下基础设施错误继续原样传播：

- `STORAGE_UNAVAILABLE`
- `STORAGE_INTEGRITY_ERROR`
- `SANDBOX_UNAVAILABLE`
- `SANDBOX_SESSION_LOST`
- `SANDBOX_CLEANUP_FAILED`

参数、路径、CAS、deny policy 等所有可预判失败必须在实际 mutation/spawn 前发生，避免 pre-effect write failure 被 ToolOperation 误判为 `EFFECT_UNKNOWN`。

验收至少覆盖：missing read、missing write parent、stale hash、edit mismatch、denied command，以及 infrastructure failure。

### F2 — 恢复 Workspace 工具模型 schema / description

恢复稳定的 workspace tool description、parameter description、return description。

优先复用 Harness public tool definition；若公开 API 不适合，则保留 LinkTools 的 declarative description，但不得复制 Harness 行为逻辑。

恢复独立 golden fixture。Golden 必须是认可的稳定 contract，不能只验证“当前 compiler contract == 当前 runtime contract”。

### F3 — Workspace Shell 行为保持统一

模型可见 Shell contract 必须与 backend 无关，统一：

- stdout/stderr 标签；
- timeout；
- background command status；
- output truncation；
- model-correctable error；
- interactive / destructive command semantic policy；
- provider API key env filtering。

`WorkspacePolicy` / workspace capability 负责 tool authorization；Local/Bubblewrap 负责 process lifecycle / OS isolation。删除导致两个 backend 模型语义不同的隐藏 policy 开关，例如 `enforce_command_policy=False`。

### F4 — Local semantic size limit 与 Bubblewrap transport limit 解耦

拆分：

1. sandbox semantic request validation；
2. Bubblewrap frame transport validation。

Local 只应用 semantic validation；Bubblewrap 再应用 frame bound。不得让 JSON framing 隐式缩小 Local API 的业务上限。

### F5 — Compaction summary model request observability 必须真实生效

不得重新实现 summarization。

使用 Pydantic AI public `WrapperModel` 包装 `SummarizingCompaction` 的 summary model request，真实接入 LinkTools `ModelRequestJournal` 与 external observer：started / completed / failed / cancelled。

Wrapper 不得改变 provider/model identity、settings、request parameters、usage accounting。当前未启用 summary streaming，因此本轮至少完整覆盖 non-streaming `request()`；未来一旦启用 streaming，必须同步覆盖 `request_stream()`。

若当前 journal 以 step index 为 active key 导致 agent request 与 compaction request 冲突，则把 active identity 改为显式 request token，不得靠覆盖规避。

### F6 — 清除 `_capabilities_native` 迁移代理与死参数

要求：

- 删除 module-level `__getattr__` 转发；
- imports 显式；
- 删除 dead args，例如已无语义的 `operation_identity_run_id`；
- static `__all__`；
- 不新增循环依赖。

按真实 owner 收敛为少量 cohesive 私有模块即可，不为了拆文件而拆文件。

### F7 — Memory `list_paths` 必须真正 bounded

`list_paths(prefix, limit=N)` 不得先扫描整个 Memory scope 再截断。

把 prefix + limit 下推到 `MemoryState.records` repository 或等价 range query，使 Harness fallback search 的 `max_search_files` 真正约束后端 I/O。FS / SQLite / MySQL / PostgreSQL 合同保持一致，不新增第二套 Memory search algorithm。

## 4. 保留但不在本轮重写的复杂度

### 4.1 Sandbox / SandboxSession

保留。它是 Bubblewrap / Local / 后续 backend 的统一隔离执行边界。

### 4.2 Local/Bubblewrap process lifecycle

保留必要实现：Windows Job Object、POSIX process group cleanup、Bubblewrap guardian/worker、runtime-death cleanup、pidfd / namespace、filelock、atomic mutation。

但不得在这些层重复维护模型 prompt/schema 或第二套 tool authorization ownership。

### 4.3 ToolOperation / Recovery

不以减少代码量为目标删除 lease、fencing、idempotency、commit/readback、effect_unknown、cancellation ownership。

## 5. 依赖

保持：

```yaml
pydantic-ai-harness>=0.29.0
```

不设置上限。不得通过 Harness `_...` 私有模块实现本轮修复。

## 6. 实施顺序

### Phase 1 — 功能降级

1. Workspace tool descriptions / golden contract；
2. Workspace `AIError -> ModelRetry` adapter；
3. Shell stdout/stderr + shared semantic policy；
4. regression tests。

完成后运行 `python manage.py check linktools-ai`。

### Phase 2 — Compaction observability

1. public `WrapperModel` observer；
2. 必要时修 journal active identity；
3. compaction request metric/trace tests。

完成后运行 `python manage.py check linktools-ai`。

### Phase 3 — 结构收缩

1. semantic validation 与 IPC frame validation 分离；
2. `_capabilities_native` 代理清理；
3. dead args 删除；
4. Memory bounded prefix listing。

完成后运行完整 master CI。

## 7. 明确禁止

本轮禁止：

- 回退 Memory / Planning / StepPersistence / Compaction 到 Native；
- 新增 workflow abstraction；
- 新增 Redis；
- 新增 DB 悲观锁；
- 为了兼容测试保留双实现；
- 通过 Harness private module patch/monkeypatch；
- 持久化 Harness DTO；
- 把 Sandbox transport framing 变成 domain persistence contract；
- 仅修改测试来适配功能降级。

## 8. 最终验收 Gate

全部满足后才可认为 PR 可合并：

1. master 原有 3 个 CI job 全绿；
2. workspace model-correctable failures 真正进入 `ModelRetry`；
3. pre-effect workspace write failure 不得进入 `EFFECT_UNKNOWN`；
4. workspace tool schema/description golden contract 恢复；
5. Local / Bubblewrap 模型侧 shell semantic policy 一致；
6. summary compaction request 在 LinkTools metrics/journal 中可观察；
7. raw transcript 始终不被 compaction 改写；
8. `MemoryStore.list_paths(limit=N)` 后端 I/O 有上界；
9. 无 `_capabilities_native.__getattr__` 动态代理；
10. 无新 public API / persistence owner 冲突；
11. Harness 依赖仍为 `>=0.29.0` 无上限；
12. `python-check.yml` / `python-publish.yml` 与 master 保持既定要求一致。
