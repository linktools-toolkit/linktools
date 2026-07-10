# linktools-cntr: manager.py / CLI 拆分设计（延续 refactor spec）

日期：2026-07-10
关联文档：`.docs/linktools-cntr-compatible-refactor-spec.md`（Phase 0-8，已完成）

## 1. 背景

`.docs/linktools-cntr-compatible-refactor-spec.md` 的 Phase 4/5 已经把 `ContainerManager`
的大部分职责拆到 `registry/`、`runtime/`、`lifecycle/`、`state/`、`repo/` 等 facade 模块下，
`ContainerManager` 本身保留为门面（原方法签名不变，内部只做委托）。

但两处仍然偏乱：

1. `linktools-cntr/src/linktools/cntr/manager.py`（469 行）里还残留三块与"编排"无关的内容：
   manager 级配置 schema 声明、chown/chmod 实现、以及一个自制的
   `_load_setting`/`_dump_setting`/`_setting_cache` 调度层。
2. `linktools-cntr/src/linktools/cntr/__main__.py`（478 行）把 4 个命令类
   （`RepoCommand`/`ConfigCommand`/`ExecCommand`/根命令 `Command`）和一个模块级单例
   `manager` 全部塞在一个文件里，不符合 `linktools-common`/`linktools-mobile` 已经在用的
   "`commands/` 下一个命令一个文件" 约定。

本次目标：延续 Phase 0-8 建立的"facade 化、签名不变、只做委托"的方式，把上述两处也理顺。
与之前几轮不同，这次不按"一次拆一个模块、多次提交"的节奏做，而是作为一次较大的改动
一次性落地（用户已确认），但仍拆成若干个逻辑独立的提交，便于回溯。

## 2. 兼容性约束（本次改动必须满足）

`ContainerManager` 不是只被本仓库调用——`linktools-cntr/src/linktools/assets/containers/**/container.py`
（内置容器）、`/workspace/projects/linktools-homelab/`、`/workspace/projects/linktools-homelab-extra/`
里的真实容器定义都通过 `self.manager.X` 直接访问它的属性/方法。已用 grep 逐一核实，
下面这份列表是当前真实存在的外部调用面，**必须原样保留（签名、行为都不能变）**：

```text
manager.change_file_owner
manager.change_file_mode
manager.containers
manager.create_docker_process
manager.create_process
manager.debug
manager.get_installed_containers
manager.project_name
manager.start_hooks
manager.user
```

以及内部（`cntr/container.py` 框架层、`registry/loader.py` 等，同仓库可控但仍需保持行为一致）
会用到的：`manager.host`、`manager.uid`、`manager.system`、`manager.docker_container_name`、
`manager.docker_compose_names`、`manager.data_path`、`manager.temp_path`、`manager.env_config`。

这次改动只允许挪动"实现"，不允许挪动/重命名/删除上面这些"外部可见"的名字。

## 3. manager.py 拆分方案

| 内容 | 现状 | 去向 | 说明 |
|---|---|---|---|
| `configs` 属性体（~40 行 `ConfigField` 声明） | `manager.py` | 新增 `cntr/config/manager.py`：`build_manager_configs(manager) -> dict` | `manager.configs` 属性保留，改为 `return build_manager_configs(self)` |
| `change_file_owner`/`change_file_mode`/`_is_chown_supported`（~55 行） | `manager.py` | 并入 `runtime/process.py` 的 `RuntimeProcessFactory`，新增 `.chown()`/`.chmod()` | 两者本来就只是调用 `create_process`，属于 runtime 的职责；manager 上的两个方法变成一行委托 |
| `_repo_path` cached_property | `manager.py` | 移入 `repo/store.py`（其唯一调用方），改为该文件内的私有辅助 | 不再是 manager 通用属性，纯粹是 repo 目录选择逻辑 |
| `_load_setting`/`_dump_setting`/`_setting_cache`/routing（~30 行） | `manager.py` | **删除**，不新建模块 | 见下方"为什么删除而不是新建 SettingsStore" |
| `data_path`/`temp_path`/`setting_path`/`root_path` | `manager.py` | **不动** | 已经是 `environ.get_data_path("container")`/`environ.get_temp_path("container")` 的一行封装，
core 已经提供了通用路径拼接逻辑，没有必要再包一层 `paths.py` |

### 为什么删除 `_load_setting`/`_dump_setting` 而不是新建一个 `SettingsStore`

追踪之后发现这一层是纯冗余：

- `ConfigStore.get()`（core, `_config.py`）直接读内存里的 `_data`（构造时/`set()` 时刚 reload 过）。
- `CacheNamespace.get()`（core, `cache.py`）每次都是即时查 SQLite，从不缓存。
- `_load_setting` 里的 `reload=True` 分支实际上**从没有**调用过底层 `ConfigStore.reload()`
  或让 `CacheNamespace` 做什么特殊操作——它只是清掉 cntr 自己那份 `_setting_cache` 字典。
  也就是说这个"reload"参数从一开始就没有让读取变得更新鲜，只是绕开了一层自制缓存。
- 三个调用方（`state/running.py` 的 `_RUNNING_KEY`、`state/installed.py` 的 `_INSTALLED_KEY`、
  `repo/store.py` 的 `_REPO_KEY`）各自都**静态**知道自己的 key 是 persistent 还是 transient
  （`_migrate.PERSISTENT_KEYS = ("INSTALLED_CONTAINERS", "INSTALLED_REPOS")`，其余走 transient），
  不需要通过一个通用调度函数去判断走哪个 store。

因此直接让三个调用方各自调用 `self.manager._persistent_store`（= `environ.config_store`）
或 `self.manager._transient_ns`（= `environ.cache.namespace("cntr")`）——这两个本来就已经是
`manager.py` 上现成的、指向 core 抽象的一行 cached_property，不需要新建任何 cntr 专属抽象。

`manager._load_running_containers`/`_dump_running_containers`（被 `lifecycle/dispatcher.py`
调用，返回的是 `BaseContainer` 对象而非名字，和 `state/running.py` 的 `RunningStateStore`
是两套不同粒度的东西，**本次不合并**——`RunningStateStore` 的模块注释明确写着
"known limitation is not fixed here to avoid mixing a behavior change with the code move"，
延续同样的克制：只把这两个方法内部改成直接调用 `self._transient_ns`，行为完全不变。

### 改动后 manager.py 的形状（示意，非最终代码）

```python
class ContainerManager:
    def __init__(self, environ, name="aio"):
        # identity: user/uid/gid/system/machine（不变）
        # environ/name/logger（不变）
        self.env_config = environ.wrap_config(...)
        self.env_config.update_defaults(**build_manager_configs(self))
        # docker_container_name/docker_compose_names（不变）
        try:
            self._migrated
        except Exception as exc:
            self.logger.warning(...)

    @property
    def configs(self):
        return build_manager_configs(self)

    # debug/container_type/container_host/host/project_name/app_path/app_data_path: 不变
    # root_path/data_path/temp_path/setting_path: 不变
    # _persistent_store/_transient_ns/_migrated: 不变（继续是 manager.py 上的 cached_property）

    # facade cached_properties（compose_runner/resolver/loader/runtime/lifecycle/
    # running_state/installed_state/repo_store）: 不变

    def change_file_owner(self, path, user, recursive=False):
        return self.runtime.chown(path, user, recursive=recursive)

    def change_file_mode(self, path, mode=0o755, recursive=False):
        return self.runtime.chmod(path, mode, recursive=recursive)

    # get_installed_containers/resolve_depend_containers/prepare_installed_containers/
    # add_installed_containers/remove_installed_containers: 不变

    def _load_running_containers(self):
        result = set()
        for name in self._transient_ns.get("RUNNING_CONTAINERS", default=[]):
            if name in self.containers:
                result.add(self.containers[name])
        return list(result)

    def _dump_running_containers(self, containers):
        self._transient_ns.set(
            "RUNNING_CONTAINERS", list(set(c.name for c in containers)))

    # notify_start/stop/remove、create_process*、get_all_repos/add_repo/update_repos/
    # remove_repo: 不变（已经是一行委托）

    # 删除：_load_setting/_dump_setting/_setting_cache/_repo_path/_is_chown_supported/
    #      change_file_owner 和 change_file_mode 的原实现
```

预计行数：469 → 约 260 行。

## 4. CLI 拆分方案：`__main__.py` → `commands/` 包

```text
cntr/commands/
  __init__.py
  _shared.py   # manager = ContainerManager(environ) 单例，
               # _iter_container_names(), _iter_installed_container_names()
  repo.py      # RepoCommand（原样搬移）
  config.py    # ConfigCommand（原样搬移）
  exec_.py     # ExecCommand（原样搬移；文件名加下划线避免与内置 exec 混淆）
  root.py      # Command：根 "cntr" 命令组（list/add/remove/up/restart/down/doctor,
               # _make_context），通过 SubCommandWrapper 挂载上面三个
```

`__main__.py` 缩成入口胶水：

```python
from .commands.root import Command

command = Command()
if __name__ == '__main__':
    command.main()
```

`manager` 单例和两个 `_iter_*_names` 辅助函数放进 `_shared.py`，而不是留在 `__main__.py` 或放进
`root.py`，是因为四个命令文件都需要它们；如果放在 `root.py`，`repo.py`/`config.py`/`exec_.py`
就要反向 import `root.py`，依赖方向是反的。

这一步纯粹是文件搬移 + import 路径调整，不改变任何命令的参数、行为，也不改变
`pyproject.toml` 里的入口点（仍然是 `linktools.cntr.__main__`），对外部 homelab 仓库零影响
（它们从不 import `linktools.cntr.__main__`）。

## 5. 兼容性与验证

- 第 2 节列出的外部调用面清单，在实现阶段作为显式 checklist 逐条核对，而不是凭记忆。
- 改动前后各跑一次 `python -m pytest tests/ -q` 全量测试，确认零回归。
- 跑一次 `python scripts/cntr_generate_snapshots.py`，确认 compose/config 输出的 snapshot
  没有漂移（结构性改动不应该改变任何生成产物）。
- 真实跑一遍 `ct-cntr list`、`ct-cntr config list`、`ct-cntr config list --show-secret`，
  确认 manager 单例构造顺序、facade 组装没有被打乱（这是本次改动的核心风险点，
  单测未必能覆盖模块导入顺序问题）。
- 虽然用户希望一次性大改，落地时仍拆成若干个逻辑独立的提交
  （manager.py 内部整理 → CLI 文件拆分），便于出问题时定位，但不分阶段验收、一次性完成。

## 6. 明确不做的事

- 不合并 `manager._load_running_containers`/`_dump_running_containers` 与
  `state/running.py` 的 `RunningStateStore`——两者语义不同（对象 vs 名字），
  合并属于行为变更，不属于本次"文件结构梳理"的范围。
- 不改变任何命令的 CLI 参数、输出格式、行为。
- 不新增 `paths.py`/`ContainerPaths` 之类的新抽象——core 已经提供了对应能力。
