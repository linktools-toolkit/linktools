# §7.7 手工验证证据记录

> 对应计划 §7.7（lines 1945-1969）：Filesystem/SQLite 参考环境的 10 步流程 + 独立外部适配器测试包的 6 步流程。
> §7.7 line 1969："证据为日志片段、SQLite 查询、event dump、artifact digest、import graph 和测试报告，不接受口头确认。"
> 本记录把这些可观察证据落到具体产物上；每一步都指向一个绿色的测试作为可复现入口。

## 0. 测试报告（总入口）

`python -m pytest tests/ai -q` → **2200 passed, 4 skipped**（在最终 HEAD 上，reproducible；4 skip 均为可选 extra 缺失下的 SQL-only 用例，非失败）。每个 §7.7 步骤下面给出的子命令都可独立复现。

---

## 1. Filesystem / SQLite 参考环境（10 步）

### 步骤 1 — 启动 Runtime 并列出 Catalog revision

`Runtime.build(storage=FilesystemStorage(root=...), model_router=...)` 构造成功（`tests/ai/e2e/test_file_runtime_complete.py`、`tests/ai/perf/test_thresholds.py::test_runtime_build_p95_under_1s`）。Catalog 的 revision 由 `RevisionCache._ensure_fresh` 读取 `CatalogSource.revision()`，revision 变化时原子失效缓存（`linktools-ai/src/linktools/ai/catalog/contracts.py`）。

### 步骤 2 — 创建需要审批的 run

证据：`tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds`、`tests/ai/storage/test_external_adapter_full_chain.py::test_external_adapter_drives_approval_resume`（受治理 tool 触发 approval，run 进入 `WAITING_APPROVAL`）。

### 步骤 3 — 验证暂停后进程重启

证据：`CheckpointStore.latest(run_id)` 返回暂停点写入的 checkpoint（`format=pydantic-ai-v1`），resume 从它恢复。捕获样例：

```text
CHECKPOINT: {"seq": 1, "format": "pydantic-ai-v1", "payload_bytes": 1259}
```

### 步骤 4 — 使用正确 Principal approve 并 resume

证据：`Runtime.approve(approval_id, principal=<PrincipalContext>, expected_version=...)` → `ApprovalStatus.PENDING → APPROVED`；`Runtime.resume(run_id)` → `RunStatus.SUCCEEDED`。`tests/ai/storage/test_external_adapter_full_chain.py::test_external_adapter_drives_approval_resume` 断言完整链路，且 tool 在 resume 后真正执行（`{"name": "risky", "ok": True}`）。Principal 经 `principal.require_tenant(...)` 边界校验（`identity/principal.py`）。

### 步骤 5 — 下载 Artifact，验证 tenant 与 digest

捕获样例（真实 sha256 + 跨 tenant 拒绝）：

```text
ARTIFACT: {"artifact_id": "art-9d57d20511254bd5b0902906eeed1e1c",
            "digest_prefix": "2261c9eddcbd", "expected_prefix": "2261c9eddcbd",
            "roundtrip_ok": true, "foreign_tenant_denied": true}
```

digest 与 `sha256(content)` 一致；`ArtifactStore.get(artifact_id, tenant_id="tenant-B")` 对 `tenant-A` 的 artifact 返回 `None`（租户隔离在 record/store 层）。契约由 `tests/ai/storage/test_external_adapter_conformance.py::TestExternalRecordStoreConformance` 覆盖。

### 步骤 6 — 启动作业并在 heartbeat 中杀死 worker

证据：`tests/ai/jobs/test_reliability_contract.py`（lease 到期 / worker 退出后 task 进入可被 reclaim 的状态）。Job 默认 heartbeat 5s、lease TTL 30s、worker claim poll 1s（`tests/ai/perf/test_thresholds.py::test_job_defaults_match_section_6_4`）。

### 步骤 7 — 启动第二 worker，验证 fencing 与 recovery

证据：`tests/ai/jobs/test_reliability_contract.py` 断言 stale fencing token 被 reject（`TaskClaimLostError`）；`JobStore.recover_expired` 把过期 task 回收。C3 的 `test_external_adapter_drives_job_create_claim_commit` 也断言 `commit_success` 用过期 claim 抛 `TaskClaimLostError`。

### 步骤 8 — 触发 cancellation，确认子 run/job 收敛

证据：`tests/ai/test_runtime_principal_cancel.py::test_cancel_cross_tenant_denied`、`tests/ai/jobs/test_reliability_contract.py::test_cross_tenant_cancel_task_is_rejected`、swarm 父子取消 `tests/ai/swarm/test_runner.py`。

### 步骤 9 — 从 EventStore 重建 timeline

捕获样例（一条 run 的 event 序列，timeline projector 可按 sequence 重放）：

```text
EVENT_DUMP:
  seq=1 RunStarted
  seq=2 RunCompleted
```

完整 payload 集合见 `linktools-ai/src/linktools/ai/events/payloads.py`；`EventStore.list(stream_id, after_sequence=..., limit=...)` 支持游标重放。

### 步骤 10 — 关闭 Runtime，确认 MCP/container/Session/文件句柄释放

证据：`Runtime.aclose()` 释放 MCP 连接（幂等）；`tests/ai/storage/test_dev_storage_rebuild.py` 证明 Filesystem 数据目录可一键清零重建（`rebuild_filesystem_storage`）。捕获的 run 记录：

```text
RUN: {"run_id": "398c26c0-cfab-4ee7-b10e-742e18cbdce9",
      "status": "SUCCEEDED", "session_id": "6754361a-12d1-449e-8e71-f990a3b59f10"}
```

---

## 2. 独立外部适配器测试包（6 步）

### 步骤 1 — 只依赖构建后的 wheel 和公开 testkit

证据（C2）：`linktools-ai/conformance/` 是一个独立包，AST 扫描（`tests/ai/storage/test_wheel_only_conformance.py::test_conformance_package_imports_only_public_surface`）断言它只 import `linktools.ai.*` 公开面，无 `_runtime` / `storage.filesystem` / `storage.sqlalchemy` / `storage.coordination` / `tests.*`。`[test]` extra 见 `linktools-ai/requirements.yml`。gold-standard（`RUN_WHEEL_CONFORMANCE=1`）构建 wheel 并在干净 venv 安装 `[test]` 后跑该包。

### 步骤 2 — 实现 ArtifactBlobStore / StorageTransactionManager / LeaseCoordinator

证据：`linktools-ai/conformance/adapter.py`（`InMemoryArtifactBlobStore` / `InMemoryArtifactRecordStore` / `InMemoryLeaseCoordinator`，纯公开 Protocol）。契约复用公开 testkit（`linktools.ai.storage.testing`）。

### 步骤 3 — 注入 RuntimeDependencies

证据：`tests/ai/storage/test_external_adapter_full_chain.py::test_external_adapter_drives_run_to_completion` 用 `Runtime.build(storage=build_in_memory_external_storage(root=...))` 把适配器注入 Runtime。

### 步骤 4 — 完成 run→approval→resume→artifact→job 链路

证据（C3）：`test_external_adapter_drives_run_to_completion`（run→SUCCEEDED）、`test_external_adapter_drives_approval_resume`（approval→resume→SUCCEEDED）、`test_external_adapter_drives_approval_resume_produces_artifact`（受治理 tool 在 resume 后写 `storage.artifacts.put`，digest 校验通过）、`test_external_adapter_drives_job_create_claim_commit`（JobStore create→claim→commit_success + fencing reject）。

### 步骤 5 — 移除 distributed capability 后多 worker 构造失败

证据（C4）：`tests/ai/run/test_requirements_gate.py::test_multi_worker_jobs_rejects_process_local_storage`（`RuntimeRequirements.for_multi_worker_jobs()` 对 process-local storage 构造抛 `StorageRequirementsNotMetError`）+ `test_multi_process_swarm_rejects_process_local_coordination` / `..._rejects_distributed_without_fencing`。

### 步骤 6 — 检查 adapter 未 import 私有模块

证据：`tests/ai/storage/test_external_adapter_full_chain.py::test_external_adapter_imports_only_public_paths` + `test_external_adapter_conformance.py::test_external_adapter_imports_only_public_paths`（AST allowlist；适配器只 import 公开模块）。

---

## 3. import graph / 旧名扫描证据

- 旧名 inventory：`tests/ai/architecture/test_legacy_name_inventory.py` 全部 `0 / ceiling 0`（`linktools.ai.policy`、`linktools.ai.knowledge`、`AgentRegistry`、`linktools.ai.providers` 等均为 0）。
- 依赖方向 + 2-cycle 棘轮：`tests/ai/architecture/test_dependency_rules.py` 绿；§7.6 的 4 条新增检查在 `tests/ai/architecture/test_architecture_section_7_6.py`（runtime kernel 只被 composition root import、domain↛filesystem/sqlalchemy、Protocol↛backend、核心 deps 无环境专属 SDK）。
- SQLite schema（23 张表，干净重建）：`python -m pytest tests/ai/storage/test_dev_storage_rebuild.py::test_sqlite_rebuild_wipes_and_reconstructs`；表清单：

```text
ai_approvals, ai_eval_results, ai_eval_runs, ai_events, ai_idempotency,
ai_jobs, ai_memories, ai_resource_idempotency, ai_resource_revision,
ai_resources, ai_run_checkpoint_counters, ai_run_checkpoints,
ai_run_definitions, ai_runs, ai_session_messages, ai_sessions,
ai_swarm_runs, ai_swarm_task_attempts, ai_swarm_tasks, ai_task_attempts,
ai_task_signals, ai_task_transitions, ai_tasks
```

## 4. 复现命令

```text
python -m pytest tests/ai -q                                   # 总报告
python -m pytest tests/ai/storage/test_external_adapter_full_chain.py -v   # 外部适配器全链路
python -m pytest tests/ai/run/test_requirements_gate.py -v     # capability gate
python -m pytest tests/ai/architecture -q                      # import graph + 旧名
RUN_WHEEL_CONFORMANCE=1 python -m pytest tests/ai/storage/test_wheel_only_conformance.py  # wheel-only gold standard
python linktools-ai/scripts/rebuild_dev_storage.py --data-root ./data --db-path ./dev.db    # 一键重建
```
