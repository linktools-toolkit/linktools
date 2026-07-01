# linktools-ai：从 sec-smartops-svc 抽取通用 Agent 运行时

## 背景

`sec-smartops-svc`（安全事件智能审计引擎）的 `engine/agent/`、`engine/capabilities/` 已经按"域无关运行时"设计，且经过近期几次重构与 secops 业务层解耦得很干净。现在需要给 A 股量化智能体项目复用同一套底座，量化项目具体设计暂不展开，本轮只解决"通用能力怎么抽"。

抽取目标仓库：`/workspace/projects/linktools`（用户现有的 Python monorepo，PyPI 已发布 `linktools` / `linktools-common` / `linktools-mobile` / `linktools-cntr`），新增子包 `linktools-ai`，遵循已有的 `{pkg}/src/linktools/{subpkg}/` + `pyproject.toml` + `tool.linktools` 元数据约定。

## 范围

**迁移**：`engine/agent/*`、`engine/capabilities/*` 中域无关的运行时代码。

**不迁移**（留在 sec-smartops-svc）：
- `EngineEnvironment`（`engine/environ.py`）—— secops 专属的运行时上下文单例（名称/logo/engine_policy/MySQL 能力目录路径等）。
- `CapabilityRepository`（`engine/capabilities/repository.py`）—— MySQL 持久化实现。
- `DbTranscriptStore`（`engine/secops/chat/transcript_store.py`）—— MySQL 会话存储实现。

这些留下的部分改为对接 `linktools-ai` 定义的 Protocol，而不是被下沉。

**现阶段不做**：`linktools-ai` 不提供 CLI 命令（不需要 `ai-xxx` 命令行工具），只作为被 import 的库；量化智能体的具体业务代码不在本次范围内。

## 架构

```
linktools-ai/src/linktools/ai/
  core/
    agent.py             BaseAgent / LlmAgent / RuntimeAgent / SubAgent
                          —— 剥离 EngineEnvironment 直接依赖，改为构造函数注入 AgentEnvironment Protocol
    session.py            FileSession / DbSession 抽象、SessionTurn 等类型（已是 Protocol 驱动，原样迁移）
    session_coordination.py / session_window.py
    stores.py              TranscriptStore / HistoryStore / ArtifactStore Protocol
                          + FileHistoryStore / FileArtifactStore 默认实现
    registry.py            AgentRegistry / SkillRegistry / SubagentRegistry / MCPRegistry
                          —— capabilities_root 路径解析等改为构造参数传入，不再隐式依赖 environ
    model_runtime.py / mcp_client.py / prompt.py / builtin_tools.py / artifact.py / skill_view.py / runtime.py
                          原样迁移（已域无关）
  capabilities/
    store.py               CapabilityStore Protocol + InMemoryCapabilityStore / FileCapabilityStore
                          （原三层缓存 mem→disk→MySQL 中去掉 MySQL 层，MySQL 实现留在 sec-smartops-svc）
    skill.py / subagent.py / mcp.py / run.py
                          原样迁移
```

### AgentEnvironment Protocol（新增，linktools-ai 定义）

从 `engine/agent/agent.py` 对 `EngineEnvironment` 的实际使用面收敛出最小契约：

- `hooks: HookRegistry | None`
- `get_logger(name: str) -> logging.Logger`
- 构建模型所需的访问入口（对应现有 `build_model(self.environ, request.model_type)` 的依赖面）

`sec-smartops-svc` 的 `EngineEnvironment` 结构化满足这个 Protocol（鸭子类型，不需要继承关系），无需包一层适配类。

## sec-smartops-svc 侧改动

- 删除 `engine/agent/`、`engine/capabilities/` 整个目录。
- `requirements.txt` 增加 `linktools-ai` 依赖。
- 全仓库 `from engine.agent import ...` / `from engine.capabilities import ...` 改为 `from linktools.ai.core import ...` / `from linktools.ai.capabilities import ...`。
- `EngineEnvironment` 补齐 `AgentEnvironment` 所需字段/方法。
- `CapabilityRepository`、`DbTranscriptStore` 改为实现 `linktools-ai` 定义的 Protocol。

## 迁移步骤

1. `linktools-ai` 建骨架：先写 Protocol + 内存/文件默认实现 + 对应单元测试（不依赖 sec-smartops-svc，可独立跑通）。
2. 逐个模块从 `engine/agent/*`、`engine/capabilities/*` 搬到 `linktools-ai`，搬一个改一处 import；`registry.py` 中唯一的全局单例耦合点（`environ.get_logger`）改为标准 `logging.getLogger(__name__)`。
3. sec-smartops-svc 补齐 `AgentEnvironment` 适配、切换 MySQL 实现挂载点。
4. sec-smartops-svc 删除旧目录、切换全部 import、加依赖。
5. 跑 sec-smartops-svc 现有测试套件（`tests/test_agent_runtime_kernel.py`、`tests/test_audit_decision_review_contract.py`、`tests/test_capability_store_sync.py` 等）作为行为不回归的验证基线。

## 测试

- `linktools-ai`：独立 pytest 套件，只测 Protocol 契约与内存/文件默认实现，不引入 MySQL/Kafka/Redis 等 sec-smartops-svc 专属依赖。
- `sec-smartops-svc`：迁移完成后跑现有测试套件确认行为不变，不新增迁移专用测试（这些模块的行为已被现有测试覆盖）。
- 量化智能体后续复用时，直接对同一套 Protocol 实现自己的存储后端（例如基于文件的回测会话存储），不需要改动 `linktools-ai` 内部。
