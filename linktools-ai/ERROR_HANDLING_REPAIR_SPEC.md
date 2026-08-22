# linktools-ai 异常体系根因闭环修复 Spec

## 1. 目标、范围与非目标

### 1.1 基线

- 仓库：`linktools-toolkit/linktools`
- 基线：`a4ded79a152ccfbc635503ee7b7bd394bd50970e`
- 实施分支：`fix/ai-error-contract-closure`
- Python 基线：`>=3.10`

### 1.2 目标

建立唯一、稳定、可追溯的异常契约，使任意生产失败满足以下规则：

1. 调用方看到的 `ErrorCode` 表示最接近根因的业务/基础设施语义，禁止由 catch 所在模块或底层存储实现决定错误码。
2. 第三方异常只在拥有该依赖的 adapter/executor 边界转换一次；上层只处理 `AIError` 和控制流异常。
3. `STORAGE_INTEGRITY_ERROR` 仅表示已证明的持久化事实损坏或 durable invariant 冲突，不作为内部断言、未知异常、超时、模型失败、资源未就绪的 fallback。
4. terminal `ExecutionResult` 自带失败信息；调用方不再二次查询 event、解析日志或猜测异常。
5. primary failure 在 terminal commit、cleanup 等二次失败发生时仍可追溯，禁止被后续异常静默覆盖。
6. `safe_error_details` 在 `AIError` 构造时即满足 JSON-safe 与敏感信息约束，错误处理路径本身不得因不可序列化 details 再次失败。

### 1.3 范围

仅修改异常产生、归一化、持久化、透传和对外展示所必需的代码与验证资产：

- `linktools.ai.errors`
- Pydantic AI Agent 执行边界
- Model registry
- Runtime execution/result/local worker/tool/subagent
- Task local launcher/service/runtime runner
- Temporal Task settle 需要的 execution failure 透传
- CLI `ai run`
- Asset/Storage 已存在的异常包装类型
- 明确被审计为错误分类的内存 invariant / 不可达断言
- 对应测试

### 1.4 非目标

- 不改变 Agent、Capability、Asset、Storage、Runtime、Temporal 的职责划分。
- 不新增 registry、strategy、provider plugin、错误配置文件、feature flag 或兼容层。
- 不修改数据库表结构和持久化 schema。
- 不为错误消息建立字符串解析规则。
- 不改变 `run()/wait()` 的控制流模型：FAILED/CANCELLED 仍返回 terminal `ExecutionResult`，不改成自动 raise。
- 不顺手重构无关模块。

## 2. 根因与目标行为

### 2.1 根因

当前异常体系同时存在以下共同根因：

1. Pydantic AI/provider exception 没有在 Agent 执行边界统一归一化，异常可直接穿透 Runtime/CLI。
2. 多个 fallback 用 `STORAGE_UNAVAILABLE`、`STORAGE_INTEGRITY_ERROR` 或 `EXECUTION_FAILED` 接住所有未知异常，导致根因被伪装。
3. `ExecutionResult` 缺少 `error_code/safe_error_details`，Task/Subagent/CLI 被迫重新解释 terminal event 或仅使用 status。
4. Task/Temporal 二次将 child execution failure 压缩成 `EXECUTION_FAILED`。
5. failure terminal 路径把“run 已建立但尚无 snapshot”的合法 interrupted 状态错误当成 storage corruption。
6. `AIError.safe_details` 未在入口保证 JSON-safe，terminal/event persistence 可能在处理原始失败时再次失败。
7. 内存 catalog/handle invariant、model route missing、公开 storage path invalid 等非存储损坏场景借用了 storage 错误码。

### 2.2 `STORAGE_INTEGRITY_ERROR` 唯一合法边界

仅以下情况可以产生 `STORAGE_INTEGRITY_ERROR`：

1. 已持久化 bytes/record 无法按照当前明确的 schema、canonical encoding 或 checksum contract 解码。
2. 已持久化引用按照 durable contract 必须指向存在对象，但目标缺失或摘要/长度不匹配。
3. 同一 durable identity 的多个已持久化事实互相矛盾，无法形成合法状态。
4. durable commit readback 已证明出现 partial commit/integrity failure。

以下情况禁止使用该错误码：模型/API 失败、等待超时、内存对象不一致、未知 Python exception、公开参数错误、资源暂未 ready、未注册 model route、正常的 interrupted execution。

## 3. 最终错误契约

### 3.1 新增并实际使用的稳定错误码

| ErrorCode | 唯一语义 | retryable |
|---|---|---:|
| `INTERNAL_ERROR` | 未被领域边界预期捕获的程序/内部运行错误；非 durable corruption | false |
| `MODEL_API_ERROR` | 模型 API 失败但没有更具体的结构化 HTTP/连接语义 | false |
| `MODEL_REQUEST_REJECTED` | 模型端明确拒绝的非 408/429 4xx 请求 | false |
| `MODEL_RATE_LIMITED` | 模型端 429 | true |
| `MODEL_TIMEOUT` | 模型请求超时或结构化 408 | true |
| `MODEL_UNAVAILABLE` | 模型连接失败或 >=500 服务不可用 | true |
| `MODEL_RESPONSE_INVALID` | Pydantic AI `UnexpectedModelBehavior` 等模型响应语义无效 | false |
| `MODEL_CONTENT_FILTERED` | Pydantic AI `ContentFilterError` | false |
| `EXECUTION_CONCURRENCY_LIMIT_EXCEEDED` | Pydantic AI execution concurrency admission 拒绝 | true |
| `EXECUTION_NOT_READY` | 请求 terminal result 时 execution 尚未 terminal | true |
| `EXECUTION_WAIT_TIMEOUT` | 调用方等待 execution terminal 超时 | true |
| `TASK_WAIT_TIMEOUT` | 调用方等待 TaskGraph terminal 超时 | true |
| `TOOL_RETRY_REQUIRED` | tool/model retry 语义需要保存在 tool operation/audit 中 | false |
| `TOOL_EXECUTION_FAILED` | tool handler 发生普通终态失败且没有更具体 `AIError` | false |
| `ASSET_NOT_FOUND` | Asset 领域资源不存在 | false |
| `STORAGE_PATH_INVALID` | Storage 公共 path/id/digest 输入不合法或越界 | false |

所有新增错误码必须至少有一个生产使用点和一个验证用例；不存在生产使用点的枚举项不得提交。

### 3.2 Pydantic AI 模型异常映射

Agent executor 只依据公开异常类型和公开字段映射，禁止读取/解析异常 message 来猜状态：

- `UsageLimitExceeded` -> `EXECUTION_USAGE_LIMIT_EXCEEDED`
- `RunCancelled` -> `EXECUTION_CANCELLED`
- `ConcurrencyLimitExceeded` -> `EXECUTION_CONCURRENCY_LIMIT_EXCEEDED`
- `ContentFilterError` -> `MODEL_CONTENT_FILTERED`
- `ModelHTTPError.status_code == 408` -> `MODEL_TIMEOUT`
- `ModelHTTPError.status_code == 429` -> `MODEL_RATE_LIMITED`
- `ModelHTTPError.status_code >= 500` -> `MODEL_UNAVAILABLE`
- 其他 `400 <= status_code < 500` -> `MODEL_REQUEST_REJECTED`
- 其他 `ModelHTTPError` -> `MODEL_API_ERROR`
- 其他 `ModelAPIError` -> `MODEL_API_ERROR`
- 其他 `UnexpectedModelBehavior` -> `MODEL_RESPONSE_INVALID`
- Pydantic/输出 `ValidationError` -> `OUTPUT_VALIDATION_FAILED`
- `UserError` 与 executor 未预期普通 `Exception` -> `INTERNAL_ERROR`
- 外部 `asyncio.CancelledError` 不得被 `Exception` 归一化逻辑吞掉，继续作为任务取消控制流传播。

`ModelHTTPError` safe details 仅允许：`model_name`、`status_code`，以及公开且为 JSON 标量的 `retry_after`；禁止 body、完整 headers、prompt、credential、原始异常文本进入 safe details。

### 3.3 `AIError.safe_details`

`AIError` 构造时递归复制并校验 `safe_details`：

- 允许 `None/bool/int/有限 float/str/list/Mapping[str, ...]`。
- Mapping key 必须为 `str`。
- 禁止 NaN/Infinity、bytes、自定义对象、异常对象和循环结构。
- 不合法立即抛 `TypeError` 或 `ValueError`，不得等到 persistence 才失败。
- `SafeError.safe_details` 与 `AIError.safe_details` 类型统一为 JSON-safe mapping。

不得自动把 `str(error)`、provider body、request/response headers、prompt、tool arguments、tool result、API key 放入 safe details。

## 4. `ExecutionResult` 公共接口

`linktools.ai.runtime.service_api.ExecutionResult` 增加：

```python
error_code: str | None = None
safe_error_details: Mapping[str, JsonValue] = field(default_factory=dict)
```

terminal invariant：

| status | error_code | safe_error_details | output/schema |
|---|---|---|---|
| `SUCCEEDED` | 必须 `None` | 必须 `{}` | 必须满足现有 success contract |
| `FAILED` | 必须为已注册 `ErrorCode` 且不得为 `EXECUTION_CANCELLED` | JSON-safe mapping | 必须无 output/schema |
| `CANCELLED` | 必须 `EXECUTION_CANCELLED` | JSON-safe mapping | 必须无 output/schema |

读取 durable terminal record 时违反该表，不做 fallback，直接 `STORAGE_INTEGRITY_ERROR`。

非 terminal execution 调用 `result()` 返回 `EXECUTION_NOT_READY`。

## 5. 文件/模块强制修改

### 5.1 `linktools-ai/src/linktools/ai/errors.py`

1. 增加第 3.1 节实际使用的错误码。
2. 更新默认 `retryable` 集合，与第 3.1 节一致。
3. 增加私有 JSON-safe details 归一化函数；不引入新模块或依赖。
4. `AIError.__init__()` 在赋值前归一化 details。
5. `SafeError.safe_details` 改为 JSON-safe mapping 类型。
6. `InvalidStoragePathError` 改用 `STORAGE_PATH_INVALID`。
7. `AssetNotFoundError` 改用 `ASSET_NOT_FOUND`。
8. 保留现有异常类名称和 public import surface；不增加兼容 alias。

### 5.2 `agent/_executor.py`

1. 在 `AgentExecutor.execute()` 边界完成第 3.2 节映射。
2. 不解析 provider 错误字符串。
3. `final_result is None` 改为 `INTERNAL_ERROR`；它不是持久化损坏。
4. usage sink 不得覆盖 primary failure：存在 primary failure 时 usage sink failure 只记录日志；不存在 primary failure 时 usage sink failure 正常传播。
5. Retry audit code 使用 `TOOL_RETRY_REQUIRED`，不再错误标成 output validation。

### 5.3 `model/_registry.py`

未注册 route 的 `resolve()` 必须抛 `MODEL_CONNECTION_NOT_FOUND`，禁止 `STORAGE_NOT_FOUND`。

### 5.4 `core/_redaction.py`

`StructuredRedactor.safe_error()` 对非 `AIError` 统一生成：

- code: `INTERNAL_ERROR`
- category: `INTERNAL`
- retryable: false
- details: `{}`
- cause digest 只使用异常类型，不使用异常文本。

### 5.5 `runtime/service_api.py` 与 `runtime/_execution.py`

1. 按第 4 节扩展 `ExecutionResult`。
2. `result()` 从 `ExecutionRecord.error_code/safe_error_details` 构造唯一 terminal failure contract。
3. durable unknown/非法 error code 视为 `STORAGE_INTEGRITY_ERROR`，禁止 fallback 为 `EXECUTION_FAILED`。
4. `wait()` timeout -> `EXECUTION_WAIT_TIMEOUT`。
5. success record 如果持有 failure code/details 视为 durable integrity error。

### 5.6 `runtime/_local.py`

1. `_task_done()` 未归一化普通异常 fallback -> `INTERNAL_ERROR`，details 固定 `{"phase": "local_execution_worker"}`。
2. `_commit_failure()` 普通异常 fallback -> `INTERNAL_ERROR`；`AIError` 原 code/details 原样持久化。
3. executor 返回/抛出 `EXECUTION_CANCELLED` 时走 CANCELLED terminal，不得记 FAILED。
4. failure terminal commit 自身失败时，新抛出的 secondary `AIError` 保留 secondary code/retryable/details，并附加：
   - `primary_error_code`
   - `primary_safe_error_details`
   Python exception cause 指向 primary failure。
5. same-storage-group FAILED/CANCELLED：
   - `run is None && snapshot is None`：合法，无 recovery run/snapshot materialization。
   - `run is not None && snapshot is None`：合法 interrupted run；不得作为 integrity failure；该 run 不作为 snapshot pair 写入 recovery archive，execution terminal seal 仍负责已有 execution projection。
   - `snapshot is not None && run is None`：`STORAGE_INTEGRITY_ERROR`。
   - 两者均存在：按现有 pair contract 写入。
6. 不降低 SUCCEEDED 对 complete snapshot 的现有强校验。

### 5.7 `runtime/_planner.py`、Task 与 Temporal

1. `RuntimeTaskNodeRunner.result()` 对 FAILED/CANCELLED 不再抛通用 `EXECUTION_FAILED`；抛与 `ExecutionResult.error_code` 对应的 `AIError`，携带 safe details；非法/缺失 durable error contract -> `STORAGE_INTEGRITY_ERROR`。
2. 增加生产需要的 terminal execution result 读取方法，仅由 Temporal Task settle 使用；不得新增 Protocol/adapter 层。
3. Local Task scheduler 未预期普通异常 fallback -> `INTERNAL_ERROR`，不再 `STORAGE_UNAVAILABLE`。
4. TaskGraph API wait timeout -> `TASK_WAIT_TIMEOUT`。
5. Temporal child execution FAILED/CANCELLED 的 node failure code 必须来自 authoritative `ExecutionResult.error_code`，禁止固定写 `EXECUTION_FAILED`。
6. Task node `error_digest` 继续只使用稳定字段，不纳入原始 exception message。

### 5.8 `runtime/_subagent.py`

Subagent tool result 增加 `error_code` 与 `safe_error_details`；值直接来自 child `ExecutionResult`。不得重新解释 status。

### 5.9 `runtime/_tool.py`

1. `ModelRetry` / `ToolRetryError` durable operation code -> `TOOL_RETRY_REQUIRED`。
2. 普通 tool exception -> `TOOL_EXECUTION_FAILED`。
3. `AIError` tool exception 必须保留其稳定 code；不得压成 `EXECUTION_FAILED`。
4. 读取 durable tool error payload 时，未知 `ErrorCode` 是 durable corruption -> `STORAGE_INTEGRITY_ERROR`，禁止 fallback `EXECUTION_FAILED`。
5. error payload 不持久化原始异常文本；只持久化稳定 code/digest 或 retry 所需、已属于模型可见协议的数据。

### 5.10 `commands/ai/run.py`

1. 删除对 `pydantic_ai.exceptions.ModelAPIError/UserError` 的 CLI 直接依赖。
2. JSON mode 直接使用 `ExecutionResult.error_code/safe_error_details`。
3. 删除 `_terminal_failure_details()` event 二次扫描。
4. streaming mode 继续读取 terminal event；event payload 与 `ExecutionResult` 使用同一 durable record 生成，禁止 CLI 自行归类。

### 5.11 其他已确认误分类

- `runtime/_agent.py` 的内存 handle/definition 不一致 -> `INTERNAL_ERROR`，不得 storage integrity。
- `agent/_catalog.py` 的同 digest 不同定义 -> `BINDING_CONFLICT`；仅内存对象结构失真 -> `INTERNAL_ERROR`。
- `agent/_definition.py` 的 definition 与 binding snapshot 自相矛盾 -> `BINDING_CONFLICT`；持久化 snapshot 的 schema/canonical 校验仍由 binding decoder 保持 `STORAGE_INTEGRITY_ERROR`。
- `adapter/_mcp.py` 删除 append 后 `len(values) != len(servers)` 的不可达 storage assertion。

## 6. 实施顺序与门禁

### 步骤 1：基础错误契约

输入：现有 `errors.py`。

动作：完成错误码、retryable 和 safe details contract。

验证：错误码可枚举；每个新增码有生产引用；非法 safe details 在 AIError 构造时失败；合法嵌套 JSON 被复制。

失败判定：存在未使用新增码、非 JSON-safe details 可进入 AIError、已有合法 AIError 构造被破坏。

下一步门禁：全部通过后进入步骤 2。

### 步骤 2：第三方执行边界

输入：步骤 1 error contract。

动作：修改 Agent executor/model registry/redactor/tool error codec。

验证：模型 HTTP 400/408/429/5xx、API error、content filter、unexpected behavior、usage、concurrency、run cancellation、generic exception 均得到唯一 code；无 message parsing。

失败判定：任何 provider exception 越过 executor；任何模型错误变为 storage code。

下一步门禁：通过后进入步骤 3。

### 步骤 3：Runtime terminal contract

动作：扩展 `ExecutionResult`，修改 result/wait/local worker/failure terminal。

验证：SUCCEEDED/FAILED/CANCELLED 三种 durable record 均满足第 4 节；non-terminal result 与 wait timeout code 正确；secondary failure 保留 primary；interrupted run 无 snapshot 可正常失败落终态。

失败判定：terminal 失败需要扫 event 才能确定错误；unknown durable code 被 fallback；failure terminal 因合法无 snapshot 产生 integrity error。

下一步门禁：通过后进入步骤 4。

### 步骤 4：下游传播

动作：修改 Task/Temporal/Subagent/CLI。

验证：child execution 的具体错误码贯穿 Local Task、Temporal Task、Subagent、CLI；任何层都不得重新写死 `EXECUTION_FAILED`。

失败判定：同一 child failure 在任一入口出现不同稳定 code。

下一步门禁：通过后进入步骤 5。

### 步骤 5：误分类清理

动作：修复 catalog/definition/handle/path/asset/MCP 已确认误分类。

验证：内存 invariant 不再产生 storage code；storage corruption 原有测试仍保持 storage integrity。

失败判定：真实 storage integrity 被改成 generic internal，或非 storage 场景仍使用 storage integrity。

## 7. 验证矩阵

必须至少覆盖以下生产行为；测试文件可按现有测试组织放置，不为测试修改生产架构。

| 场景 | 期望 |
|---|---|
| `ModelHTTPError(400)` | `MODEL_REQUEST_REJECTED`, non-retryable |
| `ModelHTTPError(408)` | `MODEL_TIMEOUT`, retryable |
| `ModelHTTPError(429)` | `MODEL_RATE_LIMITED`, retryable |
| `ModelHTTPError(500)` | `MODEL_UNAVAILABLE`, retryable |
| `ModelAPIError` | `MODEL_API_ERROR` |
| `ContentFilterError` | `MODEL_CONTENT_FILTERED` |
| `UnexpectedModelBehavior` | `MODEL_RESPONSE_INVALID` |
| `ConcurrencyLimitExceeded` | `EXECUTION_CONCURRENCY_LIMIT_EXCEEDED` |
| `RunCancelled` | terminal `CANCELLED/EXECUTION_CANCELLED` |
| generic executor exception | terminal `FAILED/INTERNAL_ERROR` |
| non-AI redaction exception | `SafeError(INTERNAL_ERROR)` |
| unknown model route | `MODEL_CONNECTION_NOT_FOUND` |
| `ExecutionResult` success | `error_code=None`, details `{}` |
| `ExecutionResult` failed | 原 durable code/details |
| `ExecutionResult` cancelled | `EXECUTION_CANCELLED` |
| unknown persisted error code | `STORAGE_INTEGRITY_ERROR` |
| execution result before terminal | `EXECUTION_NOT_READY` |
| execution wait timeout | `EXECUTION_WAIT_TIMEOUT` |
| task wait timeout | `TASK_WAIT_TIMEOUT` |
| Local worker generic failure | `INTERNAL_ERROR`，非 storage |
| failed run 无 snapshot | 正常提交 FAILED；不产生 storage integrity |
| snapshot 有而 run 缺失 | `STORAGE_INTEGRITY_ERROR` |
| failure terminal commit 再失败 | secondary code 可见，primary code/details 保留 |
| Local Task child model failure | node error code 等于 child code |
| Temporal Task child model failure | node error code 等于 child code |
| Subagent child failure | 返回 child error code/details |
| CLI JSON failed result | 不扫 event，直接输出 result error contract |
| invalid storage path | `STORAGE_PATH_INVALID` |
| missing asset | `ASSET_NOT_FOUND` |
| persisted message/canonical corruption | 仍为 `STORAGE_INTEGRITY_ERROR` |

## 8. 必须执行的验证命令

在仓库根目录执行：

```bash
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m compileall -q linktools-ai/src/linktools/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m pytest -q tests/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m ruff check linktools-ai/scripts/build linktools-ai/src/linktools/ai linktools-ai/src/linktools/commands/ai tests/ai
```

如果 CI 环境提供 MySQL/PostgreSQL contract jobs，代码改动不得跳过或降级这些现有 job；本修复不新增数据库环境要求。

## 9. Review 闭环门禁

代码完成后必须重新从基线 diff 做冷启动 Review，并同时检查：

1. 所有新增 `ErrorCode` 是否有生产使用点，是否出现重复/重叠语义。
2. 全仓新增或遗留的 `STORAGE_INTEGRITY_ERROR` 是否全部满足第 2.2 节。
3. `except Exception/BaseException` 是否吞取消、丢 primary failure、或重新误分类。
4. `ExecutionResult` 所有构造/消费者是否适配新字段。
5. Local/Temporal Task、Subagent、CLI 是否保留 child/root error code。
6. safe details 是否可序列化且不泄漏敏感信息。
7. 是否出现新 public API、抽象、配置、依赖、状态、持久化字段或兼容路径；若无当前需求直接支撑必须删除。
8. 是否存在孤儿代码、失真注释、无关重构或新增 import cycle。
9. compileall、`tests/ai`、ruff 全部通过；失败必须修复后从本节第 1 项重新复审。

仅当上述检查一整轮无已知生产问题、测试问题和证据缺口时，状态为 DONE。

## 10. 发布、监控与回滚

### 发布

本修复不含数据库迁移。以完整代码与测试门禁通过的单一 commit/PR 作为发布候选；不得拆分为“先发新 Result、后补消费者”的中间状态。

### 监控

发布后关注错误码分布；`INTERNAL_ERROR` 表示未被既有领域边界覆盖的程序问题，必须保留日志 traceback 供定位，但不对外暴露原始异常文本。

### 回滚

若发布后发现异常 contract 回归，回滚本修复的完整 commit/merge commit。由于无 schema/data migration，不执行数据回滚。已经写入的 error code 均为稳定字符串；回滚后若旧代码无法识别新 code，禁止部分回滚，必须回滚到完整发布边界并停止消费该批新 terminal records，直到恢复本修复或完成显式兼容处理。
