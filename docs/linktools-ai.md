# linktools-ai vNext 最终架构规格

## 1. 文档定位

本文档定义 linktools-ai 下一代架构和重构边界。

本次重构：

* 不考虑现有内部 API 兼容性
* 不保留旧 Agent 继承体系
* 不建立长期兼容 Adapter
* 不保留旧目录结构
* 允许重新设计数据库表和文件布局
* 以架构简洁、类型安全、可恢复、可扩展为最高目标

本文中的规范用词：

* **必须**：不可缺少
* **应当**：原则上必须满足
* **可以**：可选能力
* **禁止**：不得出现

---

# 2. 产品定位

linktools-ai 定位为：

> 基于 Pydantic AI，面向代码执行、企业安全和动态多 Agent 协作场景的模块化 Agent Runtime。

核心能力包括：

* 不可变 Agent 声明
* 可恢复的 Run 生命周期
* Swarm 动态多 Agent 协作
* 统一 Storage
* 文件与数据库两种部署模式
* 通用 Resource 管理
* Session、Memory 和 Knowledge 上下文体系
* Tool 权限、风险和审批
* Middleware 扩展机制
* 强类型事件
* 完整审计与可观测性
* Skill、MCP、Agent 配置资源化管理

当前阶段明确不实现：

* Team
* Workflow
* DAG
* 定时工作流
* Resource 原子 Batch
* 多主 Resource 写入
* Backend 双向同步
* 无限制自治 Swarm
* 完整 WebDAV 网络服务
* 大量向量数据库适配
* 完整管理控制台

---

# 3. 最终架构决策

## 3.1 只保留 Agent 和 Swarm

系统只提供两种 Runnable：

```text
Agent
Swarm
```

不引入：

```text
Team
Workflow
SubAgent 类型
Fork 类型
```

Swarm 是唯一的多 Agent 编排机制。

Swarm 内部执行的 Agent 仍然是普通 Agent，不创建特殊的 SubAgent 子类。

---

## 3.2 Agent 不持有运行状态

Agent 只是不可变声明。

以下信息不得保存到 Agent：

* Session 实例
* 当前 Run
* 当前消息
* Token 使用量
* Checkpoint 序号
* 物理工作目录
* 数据库连接
* Redis 连接
* Store 实例
* ExecutionBackend
* 当前 Swarm Task

所有运行状态归属于：

```text
Run
SwarmRun
Runtime
```

---

## 3.3 Storage 是唯一持久化入口

调用方只配置一个 Storage：

```python
storage = FileStorage(root="./data")
```

或者：

```python
storage = SqlAlchemyStorage(
    session_factory=session_factory,
)
```

Runtime 只接收：

```python
Runtime(
    storage=storage,
    models=model_router,
    execution=execution_backend,
)
```

禁止在 Agent 或 Runtime 构造函数中分别传递：

```text
session_path
memory_path
checkpoint_path
artifact_path
swarm_path
event_path
resource_path
```

---

## 3.4 统一底层能力，保留领域语义

Storage 统一：

* 文件根目录
* SQLAlchemy Engine 和 SessionFactory
* 事务
* 序列化
* 文件原子替换
* 乐观并发
* 时间戳
* 锁
* 数据库迁移
* Namespace 布局

但不将所有领域对象强行改为 Resource。

必须保留以下强类型 Store：

```text
ResourceStore
SessionStore
MemoryStore
RunStore
SwarmStore
EventStore
ApprovalStore
CheckpointStore
IdempotencyStore
```

统一的是底层实现和组装入口，不是领域接口。

---

## 3.5 数据库模式下数据库始终权威

使用 `SqlAlchemyResourceBackend` 时：

* Resource 内容以数据库为准
* Resource 状态以数据库为准
* Revision 以数据库为准
* Whiteout 以数据库为准
* 幂等记录以数据库为准
* 乐观并发条件以数据库为准

Redis 或文件协调器只用于：

* Revision 变化提示
* 可选分布式锁
* 降低数据库轮询频率
* 降低并发冲突

Redis 或协调文件不得保存 Resource 内容，也不得成为权威版本来源。

---

# 4. 总体结构

```text
Application
    │
    ▼
Runtime
    ├── AgentCompiler
    ├── AgentRunner
    ├── SwarmRunner
    ├── ToolExecutor
    ├── MiddlewarePipeline
    └── PolicyEngine
          │
          ▼
Storage
    ├── resources
    ├── sessions
    ├── memories
    └── executions
          ├── runs
          ├── swarms
          ├── events
          ├── checkpoints
          ├── approvals
          └── idempotency
```

Storage 是组合入口。

具体 Runner 不得接收完整 Storage，而只接收所需的窄接口。

---

# 5. 公共 API

根包只导出稳定的一等对象：

```python
from linktools.ai import (
    AgentSpec,
    SwarmSpec,
    Runtime,
    FileStorage,
    SqlAlchemyStorage,
    Storage,
)
```

典型初始化：

```python
storage = FileStorage(
    root="./data",
)

runtime = Runtime.build(
    storage=storage,
    models=model_router,
    execution=LocalExecutionBackend(...),
)
```

数据库模式：

```python
storage = SqlAlchemyStorage(
    session_factory=session_factory,
    resource_coordinator=RedisResourceCoordinator(redis),
)

runtime = Runtime.build(
    storage=storage,
    models=model_router,
    execution=ContainerExecutionBackend(...),
)
```

执行 Agent：

```python
result = await runtime.run(
    agent,
    "分析这个安全事件",
    session_id="session-1",
)
```

执行 Swarm：

```python
result = await runtime.run(
    swarm,
    "完成完整安全事件调查",
    session_id="session-1",
)
```

---

# 6. Runnable

AgentSpec 和 SwarmSpec 必须实现统一声明接口：

```python
class RunnableSpec(Protocol):

    @property
    def id(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...
```

Runtime 根据实际类型选择：

```text
AgentRunner
SwarmRunner
```

新增其他 Runnable 类型时，不得修改 AgentRunner。

---

# 7. Agent 架构

## 7.1 AgentSpec

```python
@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    name: str
    model: ModelPolicy
    instructions: PromptSpec
    tools: tuple[ToolRef, ...] = ()
    middleware: tuple[MiddlewareRef, ...] = ()
    output_schema: type[BaseModel] | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
```

AgentSpec 必须：

* 不可变
* 可序列化
* 可由 Markdown、YAML 或 Python 构建
* 可通过 Resource Revision 生成稳定缓存键

---

## 7.2 AgentCompiler

```python
class AgentCompiler:

    async def compile(
        self,
        spec: AgentSpec,
    ) -> CompiledAgent:
        ...
```

职责：

* 解析 ModelPolicy
* 解析 PromptSpec
* 加载 ToolRef
* 加载 Skill
* 加载 MCP Server
* 加载 Middleware
* 校验输出 Schema
* 构建 Pydantic AI Agent
* 生成编译摘要
* 缓存无状态编译结果

AgentCompiler 不负责：

* 创建 Session
* 创建 Run
* 执行模型
* 保存历史
* 创建工作目录

---

## 7.3 CompiledAgent

CompiledAgent 必须：

* 无运行状态
* 可被多个 Run 复用
* 不持有 Session
* 不持有当前 Tool Call
* 不持有 Checkpoint 状态
* 不持有当前 Workspace

---

## 7.4 AgentRunner

```python
class AgentRunner:

    async def run(
        self,
        agent: CompiledAgent,
        request: RunRequest,
        context: RunContext,
    ) -> RunResult:
        ...
```

职责：

* 加载 Session 上下文
* 加载 Memory
* 检索 Knowledge
* 执行 Middleware
* 调用模型
* 调用 ToolExecutor
* 写入 Session 消息
* 更新 Run 状态
* 发布事件
* 保存 Checkpoint
* 支持取消、暂停和恢复

禁止：

* 判断 `FileSession` 或数据库 Session 类型
* 访问物理 Session 目录
* 实例化 `LocalExecutionBackend`
* 实例化具体 Store
* 直接访问 Redis
* 从 Session 路径推导 Memory 或 Artifact 路径

---

# 8. Run 模型

## 8.1 RunContext

```python
@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    session_id: str
    runnable_id: str
    runnable_type: RunnableType
    user_id: str | None
    tenant_id: str | None
    workspace: WorkspaceRef | None
    metadata: Mapping[str, JSONValue]
```

---

## 8.2 RunStatus

```python
class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

允许的状态转换：

```text
PENDING → RUNNING

RUNNING → WAITING_APPROVAL
RUNNING → PAUSED
RUNNING → SUCCEEDED
RUNNING → FAILED
RUNNING → CANCELLED

WAITING_APPROVAL → RUNNING
WAITING_APPROVAL → CANCELLED

PAUSED → RUNNING
PAUSED → CANCELLED
```

状态转换必须由 RunStore 校验。

禁止直接修改状态字段。

---

## 8.3 RunRecord

```python
@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    root_run_id: str
    parent_run_id: str | None
    session_id: str
    runnable_id: str
    runnable_type: RunnableType
    status: RunStatus
    input: RunInput
    result: RunResult | None
    error: RunErrorInfo | None
    version: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    metadata: Mapping[str, JSONValue]
```

---

## 8.4 RunStore

```python
class RunStore(Protocol):

    async def create(
        self,
        run: RunRecord,
    ) -> RunRecord:
        ...

    async def get(
        self,
        run_id: str,
    ) -> RunRecord | None:
        ...

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: RunResult | None = None,
        error: RunErrorInfo | None = None,
    ) -> RunRecord:
        ...

    async def list_children(
        self,
        run_id: str,
    ) -> tuple[RunRecord, ...]:
        ...
```

必须提供文件和 SQLAlchemy 实现。

---

# 9. Checkpoint

Checkpoint 属于 Run，不再以 Session ID 加自增文件名表达。

```python
@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    id: str
    run_id: str
    sequence: int
    format: str
    schema_version: int
    payload: bytes
    created_at: datetime
    metadata: Mapping[str, JSONValue]
```

```python
class CheckpointStore(Protocol):

    async def save(
        self,
        checkpoint: RunCheckpoint,
    ) -> None:
        ...

    async def latest(
        self,
        run_id: str,
    ) -> RunCheckpoint | None:
        ...

    async def get(
        self,
        checkpoint_id: str,
    ) -> RunCheckpoint | None:
        ...
```

Checkpoint 必须支持：

* Agent Run 恢复
* Swarm Run 恢复
* Approval 后恢复
* 进程重启恢复

---

# 10. 统一 Storage

## 10.1 Storage 门面

```python
@dataclass(frozen=True, slots=True)
class Storage:
    resources: ResourceStore
    sessions: SessionStore
    memories: MemoryStore
    executions: ExecutionStorage
    capabilities: StorageCapabilities
```

---

## 10.2 ExecutionStorage

ExecutionStorage 是执行领域 Store 的组合命名空间，不是大型通用 CRUD 接口。

```python
@dataclass(frozen=True, slots=True)
class ExecutionStorage:
    runs: RunStore
    swarms: SwarmStore
    events: EventStore
    checkpoints: CheckpointStore
    approvals: ApprovalStore
    idempotency: IdempotencyStore
```

Runtime 可以访问完整 Storage。

AgentRunner 和 SwarmRunner 只能获得所需的子接口。

---

## 10.3 FileStorage

```python
storage = FileStorage(
    root="./data",
)
```

自动创建：

```text
FileResourceStore
FileSessionStore
FileMemoryStore
FileRunStore
FileSwarmStore
FileEventStore
FileCheckpointStore
FileApprovalStore
FileIdempotencyStore
```

调用方只配置一个根目录。

---

## 10.4 SqlAlchemyStorage

```python
storage = SqlAlchemyStorage(
    session_factory=session_factory,
    resource_coordinator=coordinator,
)
```

自动创建全部 SQLAlchemy Store。

所有 Store 必须共享：

* AsyncSession factory
* SQLAlchemy Metadata
* 数据库事务边界
* Migration 生命周期
* 数据库时间
* 乐观并发实现

禁止每个 Store 自行创建 Engine。

---

## 10.5 StorageCapabilities

```python
@dataclass(frozen=True, slots=True)
class StorageCapabilities:
    cross_store_transactions: bool
    optimistic_concurrency: bool
    append_only_events: bool
    distributed_coordination: bool
    full_text_search: bool
    semantic_search: bool
    multi_process_swarm: bool
```

文件模式默认：

```python
StorageCapabilities(
    cross_store_transactions=False,
    optimistic_concurrency=True,
    append_only_events=True,
    distributed_coordination=False,
    full_text_search=False,
    semantic_search=False,
    multi_process_swarm=False,
)
```

数据库模式：

```python
StorageCapabilities(
    cross_store_transactions=True,
    optimistic_concurrency=True,
    append_only_events=True,
    distributed_coordination=True,
    full_text_search=True,
    semantic_search=False,
    multi_process_swarm=True,
)
```

---

# 11. 数据库 Unit of Work

数据库模式必须支持跨执行领域事务：

```python
async with storage.transaction() as tx:
    await tx.approvals.create(approval)
    await tx.runs.transition(
        run_id,
        RunStatus.WAITING_APPROVAL,
        expected_version=version,
    )
    await tx.checkpoints.save(checkpoint)
    await tx.events.append(event)
```

同一个 Unit of Work 中的 Store 必须共享同一个 AsyncSession。

Resource 批量事务不在当前范围内。

文件模式不承诺跨 Store 原子事务。

文件模式依靠：

* 单文件原子替换
* Checkpoint
* EventLog
* 幂等操作
* 启动恢复

保证可恢复性。

---

# 12. FileStorage 布局

```text
data/
├── resources/
│   ├── data/
│   └── .resource/
│       ├── metadata/
│       ├── whiteouts/
│       ├── idempotency/
│       └── revision
│
├── sessions/
│   ├── records/
│   ├── messages/
│   └── summaries/
│
├── memories/
│
├── executions/
│   ├── runs/
│   ├── swarm-runs/
│   ├── swarm-tasks/
│   ├── checkpoints/
│   ├── approvals/
│   ├── idempotency/
│   └── events/
│
├── workspaces/
│
└── .storage/
    ├── metadata/
    ├── locks/
    ├── migrations/
    └── version
```

只有 FileStorage 可以了解该物理布局。

领域 Store 不得自行拼接根目录。

---

# 13. 数据库表设计

不得默认使用单一的：

```text
generic_records(namespace, key, json_payload)
```

保存全部领域数据。

应根据查询和一致性需求建立领域表：

```text
ai_sessions
ai_session_messages
ai_session_summaries

ai_memories

ai_runs
ai_run_checkpoints

ai_swarm_runs
ai_swarm_tasks

ai_approvals
ai_idempotency
ai_events

ai_resources
ai_resource_revision
ai_resource_idempotency
```

可以通过 SQLAlchemy Mixin 复用：

```text
id
tenant_id
version
created_at
updated_at
metadata
```

---

# 14. Generic ResourceStore

## 14.1 定位

ResourceStore 用于：

* Agent 定义
* Swarm 定义
* Skill
* MCP 配置
* Prompt
* 模板
* 配置文件
* Artifact
* 文本文件
* 二进制文件

ResourceStore 不理解：

* Session
* Memory
* Run 状态
* Swarm Task 状态
* Approval 状态
* Tool 执行

---

## 14.2 组合方式

ResourceStore 由：

```text
一个 Primary Backend
零个或多个只读 Overlay Backend
```

组成。

```python
store = ResourceStore(
    primary=SqlAlchemyResourceBackend(...),
    overlays=(
        FileResourceBackend(
            root="./builtin",
            readonly=True,
        ),
    ),
)
```

Primary：

* 负责全部写操作
* 保存 Whiteout
* 维护 Revision
* 保存幂等记录

Overlay：

* 默认只读
* 只参与 fallback 读取
* 不接受普通写入

禁止多个普通可写 Backend。

---

## 14.3 ResourcePath

```python
@dataclass(frozen=True, slots=True)
class ResourcePath:
    value: str
```

必须满足：

* 绝对资源路径
* 使用 `/`
* 至少包含一个 namespace
* 禁止空路径
* 禁止 `.` 和 `..`
* 禁止 NUL
* 规范化重复 `/`
* 哈希和比较基于规范化值

示例：

```text
/agents/security/agent.md
/swarms/investigation/swarm.yaml
/skills/browser/SKILL.md
/artifacts/run-123/report.json
```

---

## 14.4 Resource 模型

```python
class ResourceKind(StrEnum):
    FILE = "file"
    COLLECTION = "collection"
```

```python
@dataclass(frozen=True, slots=True)
class ResourceInfo:
    path: ResourcePath
    kind: ResourceKind
    etag: str
    version: int
    content_type: str | None
    size: int
    modified_at: datetime
    metadata: Mapping[str, JSONValue]
```

```python
@dataclass(frozen=True, slots=True)
class Resource:
    info: ResourceInfo
    content: bytes
```

提供：

```python
resource.text()
resource.json()

Resource.from_text(...)
Resource.from_json(...)
```

核心模型不得将内容固定为字符串。

---

## 14.5 查询接口

```python
await store.get(path)
await store.stat(path)
await store.propfind(
    path,
    depth=Depth.ONE,
    limit=100,
    cursor=None,
)
```

`propfind()`：

* 默认不加载内容
* 返回 ResourceInfo
* 支持分页
* 支持 Collection
* 支持 Overlay 合并

---

## 14.6 三态查找

Backend 查询结果必须为：

```python
ResourceLookup = Found | Missing | Masked
```

```python
@dataclass(frozen=True, slots=True)
class Found:
    resource: Resource
```

```python
@dataclass(frozen=True, slots=True)
class Missing:
    pass
```

```python
@dataclass(frozen=True, slots=True)
class Masked:
    path: ResourcePath
    version: int
```

查找规则：

1. 查询 Primary。
2. Found：直接返回。
3. Masked：停止查询并视为不存在。
4. Missing：继续查询 Overlay。
5. Overlay 首个 Found 返回。

---

## 14.7 Whiteout

删除任何资源都必须在 Primary 中记录 Whiteout。

即使资源只存在于 Overlay，也必须建立 Whiteout。

MOVE 必须：

1. 读取源资源。
2. 写入 Primary 目标路径。
3. 在 Primary 源路径建立 Whiteout。

这可以防止后置资源重新出现。

---

## 14.8 写入接口

```python
await store.put(path, content, options=...)
await store.delete(path, options=...)
await store.move(src, dst, options=...)
```

```python
@dataclass(frozen=True, slots=True)
class WriteOptions:
    idempotency_key: str | None = None
    if_match: str | None = None
    if_none_match: bool = False
    content_type: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    actor: str | None = None
```

---

## 14.9 幂等性

PUT 相同内容和相同元数据：

* 不创建新版本
* 不增加 Revision
* 返回现有 Resource

DELETE 不存在路径：

* 无条件删除返回成功
* 不增加 Revision

PUT、DELETE、MOVE 均支持 `idempotency_key`。

相同 Key 和相同请求哈希：

* 返回首次结果
* 不重复修改状态

相同 Key 但请求哈希不同：

* 抛出 `IdempotencyConflictError`

---

## 14.10 条件写

必须支持：

```text
If-Match
If-None-Match
```

用于：

* 乐观并发
* 防止覆盖
* 防止重复创建
* 安全 MOVE

---

## 14.11 不实现原子 Batch

当前版本不提供：

```python
apply_batch(...)
```

也不承诺：

* 多 Resource 原子修改
* 多文件事务
* 全部成功或全部失败

可以后续提供非事务型：

```python
put_many(...)
delete_many(...)
```

但必须明确每项操作独立执行、独立返回结果。

---

# 15. FileResourceBackend

必须支持：

* 只读和可写模式
* ResourcePath 安全转换
* 临时文件加原子 rename
* ETag
* 单资源版本
* Whiteout
* Revision
* 幂等记录
* 条件写
* Collection 查询
* 分页

物理路径必须经过：

```python
resolved = physical_path.resolve()
```

并验证仍位于 root 内。

符号链接策略：

```python
class SymlinkPolicy(StrEnum):
    DENY = "deny"
    ALLOW_INTERNAL = "allow_internal"
```

默认：

```text
DENY
```

FileResourceBackend 不支持跨进程强一致 Swarm。

---

# 16. SqlAlchemyResourceBackend

必须提供可直接使用的：

```python
SqlAlchemyResourceBackend(
    session_factory=session_factory,
    coordinator=coordinator,
)
```

禁止要求调用方实现：

```text
_raw_get
_raw_put
_raw_delete
_raw_move
_raw_list
```

必须支持：

* CRUD
* MOVE
* Whiteout
* ETag
* Revision
* 条件写
* 幂等键
* 乐观并发
* Collection
* 分页
* Overlay 语义

一次写操作必须在同一个数据库事务中：

1. 校验条件写。
2. 校验或预留幂等 Key。
3. 修改 Resource 或 Whiteout。
4. 更新数据库 Revision。
5. 保存幂等结果。
6. 提交事务。
7. 提交后通知 Coordinator。

---

# 17. ResourceCoordinator

数据库 Resource 可以配置一个可选 Coordinator：

```python
class ResourceCoordinator(Protocol):

    async def revision_hint(self) -> int | None:
        ...

    async def publish_revision(
        self,
        revision: int,
    ) -> None:
        ...

    def lock(
        self,
        key: str,
    ) -> AsyncContextManager[None]:
        ...
```

默认实现：

```text
RedisResourceCoordinator
FileResourceCoordinator
```

## RedisResourceCoordinator

用于：

* Revision 变化提示
* 可选分布式锁

## FileResourceCoordinator

用于共享文件系统环境：

* Revision 提示文件
* 文件锁

FileResourceCoordinator 必须明确：

* 只有所有实例共享同一可靠文件系统时才具备跨实例意义
* 不得用于保存 Resource 内容
* 不得替代数据库事务

Coordinator 故障时：

* 数据库仍然是权威
* 读操作回查数据库 Revision
* 写入正确性依赖数据库事务和约束
* 不得依赖 Coordinator 保证数据不丢失

---

# 18. Artifact

删除独立 ArtifactStore。

Artifact 是 ResourceStore 上的领域服务：

```python
class ArtifactService:

    def __init__(
        self,
        resources: ResourceStore,
    ) -> None:
        self._resources = resources
```

路径：

```text
/artifacts/{tenant_id}/{run_id}/{artifact_name}
```

ArtifactService 负责：

* 命名
* Run 关联
* 访问控制
* Content-Type
* Artifact 元数据

ResourceStore 负责：

* 内容
* ETag
* 版本
* 查询
* 幂等

---

# 19. Session

## 19.1 Session 模型

Session 是纯数据模型，不是持有 Store 和物理路径的运行对象。

```python
@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    parent_id: str | None
    status: SessionStatus
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: Mapping[str, JSONValue]
```

---

## 19.2 SessionMessage

```python
@dataclass(frozen=True, slots=True)
class SessionMessage:
    id: str
    session_id: str
    sequence: int
    role: MessageRole
    content: MessageContent
    run_id: str | None
    created_at: datetime
    metadata: Mapping[str, JSONValue]
```

禁止使用无约束字典表达核心消息字段。

---

## 19.3 SessionStore

```python
class SessionStore(Protocol):

    async def create(...)
    async def get(...)
    async def append_messages(...)
    async def list_messages(...)
    async def update(...)
    async def save_summary(...)
    async def get_summary(...)
```

必须提供：

```text
FileSessionStore
SqlAlchemySessionStore
```

不再存在：

```text
FileSession
RemoteSession
Session.copy()
Session.root
```

Swarm 子 Agent 默认创建子 Run，而不是复制 Session。

---

# 20. Memory

Memory 与 Session 必须分离。

```python
@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    owner_id: str
    content: str
    category: str | None
    confidence: float | None
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: Mapping[str, JSONValue]
```

```python
class MemoryStore(Protocol):

    async def get(...)
    async def search(...)
    async def remember(...)
    async def update(...)
    async def forget(...)
```

必须提供：

```text
FileMemoryStore
SqlAlchemyMemoryStore
```

Memory 不得通过以下方式实现：

```text
session.root / "memory"
notes.md
```

---

## 20.1 MemoryIndex

搜索能力与持久化解耦：

```python
class MemoryIndex(Protocol):

    async def index(...)
    async def remove(...)
    async def search(...)
```

文件模式可以使用关键词搜索。

数据库模式可以使用全文索引。

向量搜索作为后续可选实现。

---

# 21. Knowledge

Knowledge 不属于 Storage 中的会话数据。

```python
@dataclass(frozen=True, slots=True)
class Document:
    id: str
    content: str
    score: float | None
    source: str | None
    metadata: Mapping[str, JSONValue]
```

```python
class Retriever(Protocol):

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, JSONValue] | None = None,
        limit: int = 10,
    ) -> tuple[Document, ...]:
        ...
```

首版只要求：

```text
MemoryRetriever
```

可以增加 PgVector，但不要求大量向量数据库适配。

---

# 22. Swarm

## 22.1 定位

Swarm 是唯一多 Agent 编排机制。

用于：

* 动态任务发现
* 动态 Agent 选择
* 多轮协作
* Agent 间委派
* 并发执行
* 结果聚合
* 失败恢复

首版必须存在 Coordinator。

不实现无控制器无限自治网络。

---

## 22.2 SwarmSpec

```python
@dataclass(frozen=True, slots=True)
class SwarmSpec:
    id: str
    name: str
    agents: tuple[AgentRef, ...]
    coordinator: AgentRef
    strategy: SwarmStrategySpec
    limits: SwarmLimits
    context_policy: SwarmContextPolicy
    aggregation: AggregationPolicy
    middleware: tuple[MiddlewareRef, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
```

---

## 22.3 SwarmLimits

```python
@dataclass(frozen=True, slots=True)
class SwarmLimits:
    max_rounds: int
    max_tasks: int
    max_delegations: int
    max_depth: int
    max_concurrency: int
    max_total_tokens: int | None
    max_total_cost: Decimal | None
    timeout_seconds: float | None
```

必须防止：

* 无限轮次
* 无限任务
* 循环委派
* 无限递归
* 无限制并发
* Token 失控
* 成本失控
* 子 Run 泄漏

---

## 22.4 SwarmRun

```python
@dataclass(frozen=True, slots=True)
class SwarmRun:
    id: str
    run_id: str
    round: int
    status: SwarmStatus
    version: int
    token_usage: TokenUsage
    cost: Decimal
    created_at: datetime
    updated_at: datetime
    metadata: Mapping[str, JSONValue]
```

---

## 22.5 SwarmTask

```python
@dataclass(frozen=True, slots=True)
class SwarmTask:
    id: str
    swarm_run_id: str
    parent_task_id: str | None
    assigned_agent_id: str | None
    description: str
    status: SwarmTaskStatus
    dependencies: tuple[str, ...]
    input: TaskInput
    result: RunResult | None
    error: RunErrorInfo | None
    attempts: int
    version: int
    created_at: datetime
    updated_at: datetime
```

---

## 22.6 SwarmStore

```python
class SwarmStore(Protocol):

    async def create_run(...)
    async def get_run(...)
    async def update_run(...)
    async def create_task(...)
    async def claim_task(...)
    async def complete_task(...)
    async def fail_task(...)
    async def list_tasks(...)
```

必须提供：

```text
FileSwarmStore
SqlAlchemySwarmStore
```

FileSwarmStore：

* 只保证单进程安全
* 不声明多进程任务抢占安全
* 适用于开发和单机模式

SqlAlchemySwarmStore：

* 支持多 Worker
* 任务 Claim 必须原子
* 使用行锁或乐观条件更新
* 防止重复领取
* 支持 Lease 超时和重新领取

---

## 22.7 Swarm 与 Session

Swarm Agent 不复制 Session 对象。

所有成员共享同一个顶层 Session ID，但每次成员执行创建独立子 Run。

Context Policy 决定：

* 成员是否读取完整 Session 历史
* 成员是否只读取 Swarm 摘要
* 成员输出是否写回 Session
* 哪些输出只进入 Swarm State

默认推荐：

```text
Coordinator 读取 Session
Worker 读取任务上下文和共享摘要
最终聚合结果写回 Session
```

---

# 23. 强类型事件

## 23.1 EventEnvelope

为兼容 Python 3.11，使用 Generic：

```python
TEvent = TypeVar("TEvent")

@dataclass(frozen=True, slots=True)
class EventEnvelope(Generic[TEvent]):
    event_id: str
    sequence: int
    occurred_at: datetime
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    session_id: str
    runnable_id: str
    payload: TEvent
```

---

## 23.2 Event Payload

至少定义：

```text
RunStarted
RunCompleted
RunFailed
RunPaused
RunResumed
RunCancelled

ModelStarted
ModelCompleted
ModelFailed

ToolStarted
ToolCompleted
ToolFailed

ApprovalRequested
ApprovalApproved
ApprovalRejected

SwarmStarted
SwarmRoundStarted
SwarmRoundCompleted
SwarmTaskCreated
SwarmTaskClaimed
SwarmTaskCompleted
SwarmTaskFailed
SwarmCompleted

ResourceChanged
```

核心字段必须使用强类型。

`metadata` 可以保留为扩展字段。

---

## 23.3 EventStore

```python
class EventStore(Protocol):

    async def append(
        self,
        event: EventEnvelope[EventPayload],
        *,
        expected_sequence: int | None = None,
    ) -> EventEnvelope[EventPayload]:
        ...

    async def list(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> EventPage:
        ...
```

事件只能追加，不得覆盖。

---

# 24. Middleware

Middleware 负责生命周期横切行为：

* Budget
* Loop Detection
* Reminder
* Context Injection
* Usage Accounting
* Model Retry
* Timeout
* Checkpoint Trigger
* Logging Context

```python
class Middleware(Protocol):

    async def before_run(...)
    async def before_model(...)
    async def before_tool(...)
    async def after_tool(...)
    async def after_model(...)
    async def after_run(...)
    async def on_error(...)
```

不要求每个 Middleware 实现全部方法。

Pipeline 顺序：

```text
注册：M1, M2, M3
进入：M1 → M2 → M3
返回：M3 → M2 → M1
```

必须定义：

* 短路
* 异常传播
* 取消传播
* on_error 顺序

禁止继续在 Agent 构造函数中增加 Feature Toggle。

---

# 25. PolicyEngine

不再分别设计复杂 Guard 和 Policy 两套体系。

统一使用：

```python
class PolicyRule(Protocol):

    async def evaluate(
        self,
        request: ToolRequest,
        context: ToolContext,
    ) -> PolicyDecision:
        ...
```

```python
class PolicyDecisionKind(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
```

```python
@dataclass(frozen=True, slots=True)
class PolicyDecision:
    kind: PolicyDecisionKind
    rule_id: str
    reason: str | None
    metadata: Mapping[str, JSONValue]
```

PolicyRule 包括：

```text
PermissionRule
RiskRule
CommandRule
PathRule
NetworkRule
ResourceLimitRule
ApprovalRule
```

PolicyEngine 负责组合规则。

Middleware 不负责最终安全授权。

---

# 26. Tool

## 26.1 ToolSpec

```python
@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None
    permissions: frozenset[Permission]
    risk: RiskLevel
    approval: ApprovalMode
    side_effect: SideEffectKind
    idempotent: bool
    timeout_seconds: float | None
    metadata: Mapping[str, JSONValue]
```

---

## 26.2 ToolExecutor

负责：

* 输入 Schema 校验
* PolicyEngine 判定
* Approval
* 幂等
* 超时
* 取消
* 重试
* 事件发布
* 结果标准化
* 错误标准化

Tool 来源包括：

* Python Function
* 文件操作
* Shell
* HTTP
* MCP

Agent 间调用不作为普通 Tool 暴露，由 SwarmRunner 统一管理。

---

# 27. Tool 幂等性

```python
@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_call_id: str
    tool_name: str
    arguments: BaseModel
    idempotency_key: str | None
```

有副作用的 Tool 应支持幂等 Key。

相同 Scope、Key 和请求哈希：

* 返回首次结果
* 不重复执行副作用

相同 Key 但请求哈希不同：

* 返回冲突错误

幂等记录保存到：

```text
storage.executions.idempotency
```

---

# 28. Approval

审批不得阻塞协程等待人工输入。

```python
@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: str | None
    arguments: Mapping[str, JSONValue]
    status: ApprovalStatus
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None
```

流程：

1. PolicyEngine 返回 REQUIRE_APPROVAL。
2. 创建 ApprovalRequest。
3. 保存 Checkpoint。
4. Run 进入 WAITING_APPROVAL。
5. 当前执行返回暂停结果。
6. 外部 approve 或 reject。
7. Runtime 从 Checkpoint 恢复。

---

# 29. ExecutionBackend 与 Workspace

## 29.1 WorkspaceRef

```python
@dataclass(frozen=True, slots=True)
class WorkspaceRef:
    id: str
    run_id: str
    tenant_id: str | None
```

Agent 不得持有物理 Path。

---

## 29.2 WorkspaceManager

```python
class WorkspaceManager(Protocol):

    async def create(
        self,
        run: RunContext,
    ) -> WorkspaceRef:
        ...

    async def resolve(
        self,
        workspace: WorkspaceRef,
    ) -> ExecutionWorkspace:
        ...

    async def cleanup(
        self,
        workspace: WorkspaceRef,
    ) -> None:
        ...
```

---

## 29.3 ExecutionBackend

```python
class ExecutionBackend(Protocol):

    async def run_bash(...)
    async def read_file(...)
    async def write_file(...)
    async def list_dir(...)
    async def terminate(...)
```

默认实现：

```text
LocalExecutionBackend
ContainerExecutionBackend
```

必须支持：

* 工作目录限制
* 路径限制
* 环境变量白名单
* 命令超时
* 输出大小限制
* 进程取消
* 网络策略
* 资源限制

AgentRunner 不得实例化具体 Backend。

---

# 30. Registry

依赖方向：

```text
Storage.resources
    ↓
Registry / Parser
    ↓
AgentSpec / SwarmSpec / SkillSpec / MCPServerSpec
    ↓
Compiler / Runtime
```

Registry 负责：

* 加载 Resource
* 解析 Markdown、YAML、JSON
* Schema 校验
* ID 和名称索引
* Revision 感知
* 编译缓存失效

Registry 不负责：

* Resource 持久化
* Run 执行
* Session 存储
* Tool 执行

建议提供：

```text
AgentRegistry
SwarmRegistry
SkillRegistry
MCPRegistry
ToolRegistry
```

---

# 31. Model 层

```text
ModelRegistry
ModelPolicy
ModelRouter
```

```python
@dataclass(frozen=True, slots=True)
class ModelPolicy:
    primary: str
    fallbacks: tuple[str, ...] = ()
    max_retries: int = 0
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    budget: Decimal | None = None
```

ModelRouter 负责：

* 主模型选择
* Fallback
* 重试
* Provider 错误分类
* 超时
* Token 限制
* 预算限制

Fallback 不得只是未使用配置字段。

---

# 32. 错误模型

必须定义稳定的领域错误：

```text
LinktoolsAIError

RunError
RunNotFoundError
RunConflictError
RunCancelledError
InvalidRunTransitionError

ToolError
ToolDeniedError
ToolApprovalRequiredError
ToolTimeoutError
ToolIdempotencyConflictError

ResourceError
ResourceNotFoundError
ResourceConflictError
ResourcePreconditionFailedError
ResourceReadOnlyError
ResourceUnsupportedError
InvalidResourcePathError

SessionError
MemoryError
RegistryError

SwarmError
SwarmLimitExceededError
SwarmTaskConflictError

StorageError
StorageCapabilityError
IdempotencyConflictError
```

禁止通过错误字符串匹配类型。

---

# 33. 推荐目录结构

```text
linktools/ai/
├── agent/
│   ├── spec.py
│   ├── compiler.py
│   ├── runner.py
│   └── models.py
│
├── run/
│   ├── context.py
│   ├── models.py
│   ├── state.py
│   └── checkpoint.py
│
├── swarm/
│   ├── spec.py
│   ├── runner.py
│   ├── strategy.py
│   ├── models.py
│   ├── limits.py
│   └── aggregation.py
│
├── session/
│   ├── models.py
│   ├── store.py
│   └── context.py
│
├── memory/
│   ├── models.py
│   ├── store.py
│   ├── index.py
│   └── manager.py
│
├── knowledge/
│   ├── document.py
│   ├── retriever.py
│   └── context.py
│
├── tool/
│   ├── spec.py
│   ├── handler.py
│   ├── executor.py
│   ├── registry.py
│   ├── result.py
│   ├── builtin/
│   └── mcp/
│
├── model/
│   ├── registry.py
│   ├── policy.py
│   ├── router.py
│   └── factory.py
│
├── middleware/
│   ├── base.py
│   ├── pipeline.py
│   ├── budget.py
│   ├── loop_guard.py
│   ├── reminder.py
│   ├── checkpoint.py
│   └── context.py
│
├── policy/
│   ├── engine.py
│   ├── rule.py
│   ├── permission.py
│   ├── risk.py
│   ├── command.py
│   ├── path.py
│   ├── network.py
│   └── approval.py
│
├── execution/
│   ├── backend.py
│   ├── workspace.py
│   ├── local.py
│   └── container.py
│
├── events/
│   ├── envelope.py
│   ├── payloads.py
│   └── models.py
│
├── registry/
│   ├── parser.py
│   ├── agent.py
│   ├── swarm.py
│   ├── skill.py
│   ├── mcp.py
│   └── tool.py
│
├── storage/
│   ├── facade.py
│   ├── capabilities.py
│   ├── transaction.py
│   │
│   ├── resource/
│   │   ├── path.py
│   │   ├── models.py
│   │   ├── lookup.py
│   │   ├── store.py
│   │   └── artifact.py
│   │
│   ├── file/
│   │   ├── storage.py
│   │   ├── resource.py
│   │   ├── session.py
│   │   ├── memory.py
│   │   ├── run.py
│   │   ├── swarm.py
│   │   ├── event.py
│   │   ├── approval.py
│   │   └── idempotency.py
│   │
│   ├── sqlalchemy/
│   │   ├── storage.py
│   │   ├── unit_of_work.py
│   │   ├── resource.py
│   │   ├── session.py
│   │   ├── memory.py
│   │   ├── run.py
│   │   ├── swarm.py
│   │   ├── event.py
│   │   ├── approval.py
│   │   ├── idempotency.py
│   │   └── models/
│   │
│   └── coordination/
│       ├── base.py
│       ├── redis.py
│       └── file.py
│
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   └── logging.py
│
├── runtime.py
├── errors.py
└── __init__.py
```

禁止重新建立：

```text
core/
support/
team/
workflow/
subagent/
fork/
```

---

# 34. Python 版本

最低 Python 版本调整为：

```toml
requires-python = ">=3.11"
```

代码必须以 Python 3.11 语法为基线。

如使用 Python 3.12 专属语法，必须将最低版本同步调整为 3.12。

---

# 35. 必须删除的现有结构

直接删除：

```text
BaseAgent / LlmAgent / RuntimeAgent / SubAgent 继承体系
Feature Toggle 构造参数
Session.root
FileSession / RemoteSession 分支逻辑
Session.copy()
RunStatusStore 位于 session 模块的设计
MemoryCapability 的路径文件实现
TaskQueue 作为 Swarm 核心抽象
FileTaskQueue 整体 tasks.json 实现
Fork
Subagent
独立 AgentArtifactStore
Checkpoint 按 session_id + seq 存储的旧接口
DatabaseBackend._raw_*
Resource apply_batch()
core/
support/
registry_store/
```

现有 Swarm 仅保留需求，不保留实现。

---

# 36. 实施阶段

## 阶段一：Storage 与 Resource

完成：

* Storage 门面
* FileStorage
* SqlAlchemyStorage
* ResourcePath
* Resource 模型
* Primary + Overlay
* Whiteout
* ETag
* 条件写
* 幂等
* FileResourceBackend
* SqlAlchemyResourceBackend
* RedisResourceCoordinator
* FileResourceCoordinator
* ArtifactService

## 阶段二：Run、事件和 Session

完成：

* RunContext
* RunRecord
* Run 状态机
* FileRunStore
* SqlAlchemyRunStore
* CheckpointStore
* 强类型 Event
* FileEventStore
* SqlAlchemyEventStore
* FileSessionStore
* SqlAlchemySessionStore

## 阶段三：Agent Runtime

完成：

* AgentSpec
* AgentCompiler
* AgentRunner
* Runtime
* ModelRouter
* ToolExecutor
* MiddlewarePipeline
* PolicyEngine
* WorkspaceManager

完成后删除旧 Agent 体系。

## 阶段四：Swarm

完成：

* SwarmSpec
* SwarmRunner
* CoordinatorStrategy
* SwarmRun
* SwarmTask
* FileSwarmStore
* SqlAlchemySwarmStore
* 子 Run
* 动态任务分配
* 并发控制
* 失败恢复
* Token 和 Cost 限制

## 阶段五：Memory 与 Knowledge

完成：

* FileMemoryStore
* SqlAlchemyMemoryStore
* MemoryIndex
* MemoryManager
* Retriever
* Context Builder
* Summary 和 Compression

## 阶段六：企业能力

完成：

* ApprovalStore
* 持久化审批
* OpenTelemetry
* Metrics
* 审计接口
* 可选 ASGI API

---

# 37. 测试要求

## 37.1 Storage 与 Resource

必须覆盖：

* Primary 优先
* Overlay fallback
* Whiteout 防止资源复活
* 删除 Overlay 资源
* MOVE Overlay 资源
* 路径逃逸
* 符号链接逃逸
* PUT 相同内容不增加 Revision
* 条件 PUT
* 条件 DELETE
* 幂等 PUT
* 幂等 DELETE
* 幂等 MOVE
* 幂等冲突
* Coordinator 故障
* 数据库 Revision 权威
* Propfind 分页
* Collection 查询
* 文件单资源原子替换

## 37.2 Run 与 Session

必须覆盖：

* Run 状态转换
* 非法状态转换
* 乐观并发冲突
* FileStore 重启恢复
* SQLAlchemy 并发更新
* 父子 Run
* Checkpoint 保存和恢复
* Session 消息顺序
* Session 分页
* Session Summary
* Session 不依赖物理路径

## 37.3 Agent

必须覆盖：

* AgentSpec 不可变
* Compiler 引用校验
* Compiler 缓存失效
* Middleware 顺序
* Middleware 短路
* Model Fallback
* Tool Schema 错误
* Tool Timeout
* Tool Cancel
* Policy Deny
* Approval Pause
* Approval Resume
* Tool 幂等
* Run Cancel
* Checkpoint Resume

## 37.4 Swarm

必须覆盖：

* Coordinator 创建任务
* 动态 Agent 选择
* 多轮执行
* 任务依赖
* 原子 Claim
* Lease 超时
* 任务重试
* 最大轮次
* 最大任务数
* 最大委派数
* 最大深度
* 循环委派检测
* 最大并发
* Token 限制
* Cost 限制
* Worker 失败
* Swarm 暂停
* 进程重启恢复
* 子 Run 关系
* 聚合结果
* 取消传播
* 文件模式拒绝多进程 Swarm

---

# 38. 最终验收标准

完成后必须满足：

1. Agent 不持有运行状态。
2. 不存在旧 Agent 继承体系。
3. 不存在 Team。
4. 不存在 Workflow。
5. 不存在 SubAgent 类型。
6. Swarm 是唯一多 Agent 编排机制。
7. 不存在 Agent Feature Toggle。
8. Runtime 只接收一个 Storage。
9. AgentSpec 和 AgentRunner 不接收物理路径。
10. 文件模式只配置一个 root。
11. 数据库模式只配置一个 SessionFactory。
12. 所有 SQLAlchemy Store 可以共享事务。
13. Runtime 不把完整 Storage 传入 Runner。
14. Runner 只接收所需的领域 Store。
15. Session、Memory、Run、Swarm 保留强类型领域接口。
16. ResourceStore 不吞并 Session、Memory、Run 或 Swarm。
17. ResourceStore 只有一个主写 Backend。
18. Overlay 默认只读。
19. Whiteout 阻止资源复活。
20. ResourcePath 不允许路径逃逸。
21. SqlAlchemyResourceBackend 可直接使用。
22. 不存在 `_raw_*` 数据库抽象。
23. 不存在 Resource 原子 Batch。
24. 数据库是数据库 Resource 的唯一权威数据源。
25. Redis 和文件 Coordinator 只用于提示与锁。
26. Coordinator 故障不影响数据正确性。
27. PUT、DELETE 和 MOVE 支持幂等键。
28. Tool 支持幂等执行。
29. Artifact 基于 ResourceStore。
30. Session 不持有 Store 或 root。
31. Memory 与 Session 分离。
32. Run 具有明确状态机。
33. Run、Session、Memory、Swarm 均提供文件和数据库实现。
34. Swarm 数据库模式支持多 Worker 原子领取任务。
35. Swarm 文件模式明确限制为单进程。
36. Swarm 支持恢复。
37. 所有核心事件具有强类型 Payload。
38. Middleware 与 PolicyEngine 职责分离。
39. AgentRunner 不实例化 ExecutionBackend。
40. Workspace 不以 Path 暴露给 Agent。
41. 根包具有有限公共 API。
42. Python 最低版本与实际语法一致。
43. 不存在 `core/` 和 `support/`。
44. 核心文件原则上不超过 400 行。
45. 新增 Tool 不修改 AgentRunner。
46. 新增 Middleware 不修改 AgentSpec 字段。
47. 新增 Resource Backend 不修改 ResourceStore 核心逻辑。
48. 新增 SwarmStrategy 不修改 AgentRunner。
49. README 与实际实现一致。
50. 文件和数据库模式通过同一套领域契约测试。

---

# 39. 最终原则

最终实现必须遵循：

> Agent 只负责声明，Run 承担状态，Swarm 负责动态协作，Runtime 负责组装执行，Storage 统一底层持久化，领域 Store 保留业务语义，Resource 只管理资源，数据库始终权威，协调组件只做优化，所有运行都可恢复，所有关键事件都强类型。

最终目标是构建一个：

* 架构清晰
* 类型安全
* 模块边界稳定
* 存储配置简单
* 可恢复
* 可审计
* 强权限
* 易于扩展
* 适合企业代码执行和安全调查场景

的 Agent Runtime。
