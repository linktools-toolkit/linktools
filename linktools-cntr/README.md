# linktools-cntr

Docker 容器部署和管理工具，为 homelab 及服务器环境提供统一的容器生命周期管理（命令前缀 `ct-cntr`）。

## 开始使用

以基于 Debian 的系统为例，先安装运行环境：

```bash
# 安装 Python3、Git、Docker、Docker Compose
wget -qO- get.docker.com | bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip git docker-compose-plugin
```

安装 linktools-cntr：

```bash
python3 -m pip install -U linktools linktools-cntr

# 安装 GitHub 最新开发版
python3 -m pip install --ignore-installed \
  "linktools@ git+https://github.com/linktools-toolkit/linktools.git@master#subdirectory=linktools" \
  "linktools-cntr@ git+https://github.com/linktools-toolkit/linktools.git@master#subdirectory=linktools-cntr"
```

## 容器部署示例

### All in one 环境

PVE、OpenWRT、飞牛 OS、WAF、SSO、导航页等等

👉 [搭建文档](https://github.com/linktools-toolkit/linktools-homelab/blob/master/2xx-homelab/221-fnos/README.md)

### Xray Server

gRPC + SSL + VLESS

👉 [搭建文档](https://github.com/linktools-toolkit/linktools-homelab/blob/master/3xx-proxy/320-xray-server/README.md)

### Redroid

Docker 版 Android 容器，以及编译环境

👉 [搭建文档](https://github.com/linktools-toolkit/linktools-homelab/blob/master/4xx-mobile/400-redroid/README.md)

## 内置容器

linktools-cntr 内置了常用容器定义，开箱即用：

| 容器 | 说明 |
|------|------|
| nginx | 反向代理（含 ACME 自动证书） |
| lldap | 轻量级 LDAP 目录服务 |
| authelia | 单点登录 / 双因素认证 |
| safeline | Web 应用防火墙 |
| portainer | 容器可视化管理界面 |

更多容器可通过添加外部仓库获取（参见下方仓库管理命令）。

## 内置配置项

首次部署时会引导填写配置项，内置的全局配置参数包括：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `CONTAINER_TYPE` | str | — | 容器运行时：`docker` / `docker-rootless`（Podman 已不再支持） |
| `DOCKER_USER` | str | 当前用户 | 部分 rootless 容器使用此用户权限运行 |
| `DOCKER_HOST` | str | `/var/run/docker.sock` | Docker Daemon 地址 |
| `DOCKER_APP_PATH` | str | `~/.linktools/data/container/app` | 容器数据持久化目录（建议置于 SSD） |
| `DOCKER_APP_DATA_PATH` | str | 默认同`DOCKER_APP_PATH` | 不频繁读写的持久化目录（可置于 HDD） |
| `HOST` | str | 当前局域网 IP | 宿主机 IP 地址 |

## 常用命令

```bash
# 查看帮助（每个子命令均支持 -h 参数）
ct-cntr -h

#######################
# 仓库管理（支持 git 链接和本地路径）
#######################

# 添加容器仓库
ct-cntr repo add https://github.com/linktools-toolkit/linktools-homelab

# 拉取仓库最新代码
ct-cntr repo update

# 删除仓库
ct-cntr repo remove

#######################
# 容器安装列表管理
#######################

# 添加要部署的容器
ct-cntr add nginx lldap authelia portainer

# 从部署列表移除容器
ct-cntr remove nginx

#######################
# 容器生命周期管理
#######################

# 启动容器
ct-cntr up

# 重启容器
ct-cntr restart

# 停止容器
ct-cntr down

#######################
# 配置管理
#######################

# 查看 linktools-cntr 自身配置的帮助（Docker Compose 配置见下方 ct-cntr compose）
ct-cntr config

# 列出所有配置变量
ct-cntr config list

# 设置配置变量
ct-cntr config set NGINX_ROOT_DOMAIN=example.com ACME_DNS_API=dns_ali Ali_Key=xxx Ali_Secret=yyy

# 指定 ACME CA 与账户邮箱（默认 CA 为 Let's Encrypt，邮箱可选）
ct-cntr config set ACME_SERVER=letsencrypt ACME_ACCOUNT_EMAIL=admin@example.com

# 删除配置变量
ct-cntr config unset NGINX_ROOT_DOMAIN ACME_DNS_API Ali_Key Ali_Secret

# 使用编辑器直接编辑配置文件
ct-cntr config edit --editor vim

# 重新加载配置
ct-cntr config reload
```

nginx 的 ACME 证书在镜像**构建期**签发。证书域名或 CA 等构建输入变化时，会生成新的 nginx 镜像标签并在部署前构建；容器启动前只离线校验证书、导入镜像中已签发的证书与 ACME 状态，不进行网络签发。当前有效证书会继续复用，新增 SAN 则通过 `certs/versions` 与 `certs/live` 原子切换，原有证书和账号数据不会直接覆盖。自动续期仍在容器运行期间按照 cron 执行；证书域名或 CA 更新将触发新镜像构建。镜像构建会显式重新签发证书，因此反复强制重建可能触发 CA 频率限制。`--pull` 只请求更新基础镜像，不保证跳过 Docker 构建缓存或重新签发证书。

## 进阶功能

```bash
#######################
# 输出最终解析后的 Docker Compose 模型（只读；不涉及生命周期）
#######################

ct-cntr compose                        # 完整已安装项目
ct-cntr compose nginx --format json    # 只筛选 nginx 对应的 service
ct-cntr compose --check                # 只校验，不输出内容

#######################
# 实际运行状态（只读；如需要 sudo 密码会阻塞等待输入）
#######################

ct-cntr status
ct-cntr status --json

#######################
# 执行计划（只展示会发生什么，不实际执行）
#######################

ct-cntr up --dry-run
ct-cntr restart --dry-run
ct-cntr down --dry-run

#######################
# 诊断（只读；--json 输出结构化结果供 CI 使用）
#######################

ct-cntr doctor --json
ct-cntr doctor --check        # 存在 WARN 级别 finding 时非零退出
ct-cntr doctor --runtime      # 额外对实际 docker/compose 运行时校验 compose config
```

### 本地文件配置（`.linktools.json`）

`.linktools.json`/`linktools.json` 是通用的项目清单（project profile，`linktools.core.ProjectProfile` 负责读取与合并），不是 cntr 专属格式，也不是独立的配置系统——它只是接入现有 ConfigResolver 的两个轻量文件层：用户级 `~/.linktools/linktools.json` 与本地级 `<root>/.linktools.json`。不要求 `version`/`kind`/`schema_version`/`components`。容器仓库可以在根目录放置本地文件，声明该仓库对 `linktools-cntr` 的版本要求（`requires`），以及仓库内容器的本地默认环境值（`env`）。缺失该文件的仓库正常可用，行为不变。

```json
{
  "requires": {
    "linktools-cntr": ">=0.10.0,<1.0"
  },
  "env": {
    "STORAGE_PATH": "./storage"
  }
}
```

cntr 只读取仓库自己本地文件里的 `requires.linktools-cntr`——用户级文件、`ct-cntr config set` 持久化值、运行时覆盖都不能放宽或覆盖仓库自身声明的兼容性要求。不满足（或 specifier 非法）时，`repo add`/`repo update`/加载都会在该仓库的 `container.py` 被导入前拒绝；`requires` 中的其他 key（如未来的 `linktools-ai`）cntr 完全忽略。

### Compose 渲染后 Hook

在容器的 `on_init()` 中通过 `self.hooks.register(HookPhase.AFTER_COMPOSE_RENDER, ...)` 注册 Hook。模板解析及服务、网络默认值补全后，Hook 按 `order`、依赖关系和注册顺序执行；接收渲染后的 Compose 字典，可原地修改。每个容器实例首次渲染时执行一次；没有 Compose 模板时不执行。

```python
from typing import TYPE_CHECKING
from linktools.cntr import BaseContainer
from linktools.cntr.lifecycle import HookPhase

if TYPE_CHECKING:
    from typing import Any

class Container(BaseContainer):
    def on_init(self) -> None:
        self.hooks.register(
            HookPhase.AFTER_COMPOSE_RENDER,
            self._add_compose_labels,
            key="compose-labels",
        )

    def _add_compose_labels(self, compose: "dict[str, Any]") -> None:
        compose["services"]["app"].setdefault("labels", {})["example.enabled"] = "true"
```

修改会体现在后续缓存、Compose 文件和执行计划中。`ct-cntr doctor`、执行计划等只读操作也可能触发渲染，因此此 Hook 只应修改传入的内存字典，不应写文件或执行其他外部操作。

迁移表：

- `ct-cntr plan up` → `ct-cntr up --dry-run`
- `ct-cntr plan restart` → `ct-cntr restart --dry-run`
- `ct-cntr plan down` → `ct-cntr down --dry-run`

旧 JSON 输出分别改为追加 `--json`。`--json` 必须与 `--dry-run` 同时使用。

```bash
ct-cntr repo status
ct-cntr repo validate --json
ct-cntr repo update --json   # 每个仓库都会更新并重新校验；任意仓库更新失败或不兼容都会让命令非零退出
```

## 容器操作时序

共享上下文为 `OperationContext`。容器使用已有的准备与检查回调，不再实现一套生成配置生命周期。

`up`：确定范围 → `on_starting / BEFORE_START` → 准备镜像 → `on_check / CHECK` → 框架应用服务并确认就绪 → `on_started / AFTER_START`。

`restart` 在准备和检查全部通过后才停止显式目标；依赖方不进入显式停止集合。`down` 不准备启动配置或密钥。状态查询不调用有副作用的准备回调。

准备阶段通过 `context.write_files(self, files)` 写入不可变候选文件；检查阶段验证相同输入。框架依据实际文件挂载和 Compose 模型决定哪些服务需要重建，并统一记录应用结果与恢复旧模型。普通配置部署不再对 Nginx 执行热加载；ACME 续期仍由证书脚本负责 reload。

容器组依赖用于选择参与服务，真正的启动先后由 Compose `depends_on` 等依赖决定。实际依赖环直接报错，不再自动使用 Bootstrap 配置。配置检查失败不停止旧服务；后置通知失败明确报告“已应用、后置处理失败”，不反向触发部署回滚。

详见 [生命周期与文件准备](docs/lifecycle.md)。

## 相关链接

- GitHub: <https://github.com/linktools-toolkit/linktools/tree/master/linktools-cntr>
- homelab 容器仓库示例: <https://github.com/linktools-toolkit/linktools-homelab>

## 声明式集成与配置发布

`integrations` 返回扁平的 `Integration` 数组，通过 `Nginx` 和 `Flare` 工厂统一声明：

```python
from linktools.cntr import Flare, Nginx

return [
    Nginx.site("app.example.com", link=Flare.public("应用", "web", "应用描述")),
    Flare.bookmark("工具", "web", "https://tool.example.com", category="tool"),
]
```

`Flare.public` 创建带描述的应用；`Flare.container(name, icon, url)` 创建容器分区书签；
`Flare.bookmark` 支持自定义分区。
使用 `Flare.category("tool", "工具", order=5)` 可进一步设置分区标题和顺序。
域名配置使用 `ConfigField(provider=Nginx.domain(self))`。
共享声明位于 `integration/` 包中；容器在已有准备、检查回调中处理自己的输入，
服务应用、文件变更判断和恢复统一由框架负责。
导航 URL 不再负责注册代理。
`auth_bypass` 与 `waf_bypass` 分别控制认证和 WAF 路径旁路；自定义模板保留 nginx 原生路由语义。
外部容器仓库需要同时迁移 Python 声明、模板和 OIDC 读取接口。
详见 [集成协议与迁移说明](docs/integrations.md)。

### Nginx 与 SafeLine 网络

SafeLine 独立拥有 `safeline-ce` 网络，Tengine 和其他 SafeLine 服务只加入这个网络。
启用 SafeLine 时，Nginx 额外加入该网络，固定为 `.253`；Tengine 固定为 `.254`。
Nginx 保留自己的应用网络，但 Tengine 不加入它。网段沿用 `SAFELINE_SUBNET_PREFIX`，默认 `172.22.242`。

回源契约仍是 `http://nginx:<NGINX_WAF_PORT>`，默认 `http://nginx:8000`，已有正确配置无需改名。
固定地址避免容器重建后双向代理仍使用旧 IP；两者不共享网络命名空间，Nginx 健康检查不等待 Tengine。
回源端口只走业务/认证路由，不再次进入 WAF，也不跳转 HTTPS；SafeLine 必须保留原始 Host 和 `X-Proxy-Original-*` 头。
该端口不发布到宿主机，并只信任 Tengine 的精确地址。不能把回源指向 Nginx 的公开入口，否则会形成循环。
详见 [网络与回源契约](docs/integrations.md#nginx-and-safeline-networks)。
