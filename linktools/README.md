# Linktools

Linktools 核心框架，提供命令行工具基础设施、环境管理及通用工具集。

## 开始使用

### 依赖项

Python & pip（3.6 及以上）：<https://www.python.org/downloads/>

### 安装

```bash
# 安装核心包
python3 -m pip install -U linktools

# 安装完整功能（包含所有可选依赖）
python3 -m pip install -U "linktools[all]"

# 安装 GitHub 最新开发版
python3 -m pip install --ignore-installed \
  "linktools@ git+https://github.com/linktools-toolkit/linktools.git@master#subdirectory=linktools"
```

### 配置 alias（推荐）

对于 *nix 系统，推荐在 `~/.bashrc` 或 `~/.zshrc` 中添加以下配置：

```bash
# 生成 alias 脚本，简化命令调用（如果 PATH 未正确配置，或使用 venv 安装时特别有用）
eval "$(python3 -m linktools.cli.env alias --shell bash)"
```

## 目录结构

linktools 核心包目录：

```
linktools/src/linktools/
├── core/           — 核心模块：environ、Config、Tool、Capability
├── cli/            — 命令行框架：BaseCommand、BaseCommandGroup、CommandParser
├── utils/          — 通用工具函数：类型转换、文件、网络、进程、端口等
├── assets/         — 静态资源：工具定义（tools.yml）等
├── references/     — 内联的第三方库，避免引入额外依赖（如 fake_useragent）
├── decorator.py    — 常用装饰器：@singleton、@cached_property、@try_except 等
├── types.py        — 公共类型与异常体系：Timeout、Stoppable、Error 等
└── rich.py         — 终端 UI：日志、进度条、prompt/confirm/choose
```

各子包在此基础上扩展以下目录：

```
{subpackage}/src/linktools/
├── assets/         — 子包静态资源：配置模板、内置脚本、Agent 等
├── commands/       — CLI 命令实现，按功能分子目录（common / android / ios 等）
└── capabilities/   — 子包能力声明，向核心框架注册 root_path 和版本信息
```

## 主要功能

### 命令行框架

提供统一的命令行工具基础设施，所有子包的命令均基于此框架构建：

- `BaseCommand` — 单命令基类
- `BaseCommandGroup` — 命令组基类（支持子命令）
- `CommandParser` — 增强版 ArgumentParser，支持配置文件集成

### 统一入口

```bash
# 查看所有已安装的命令
$ lt
# 或
$ python3 -m linktools
```

### 扩展性

linktools 通过 Python entry points 机制加载各子包命令，安装对应子包后命令自动注册：

| 子包 | 命令前缀 | 说明 |
|------|----------|------|
| `linktools-common` | `ct-` | 通用工具命令 |
| `linktools-mobile` | `at-` / `it-` | Android / iOS 设备命令 |
| `linktools-cntr` | `ct-cntr` | 容器管理命令 |

## Python API

### environ — 环境管理

全局单例 `environ`，统一管理数据目录、缓存路径、日志及配置：

```python
from linktools.core import environ

# 数据/缓存目录
environ.data_path          # 持久化数据目录
environ.temp_path          # 临时文件目录
environ.get_data_path("subdir", "file.txt")  # 拼接子路径
environ.get_temp_path("cache")

# 日志
logger = environ.get_logger(__name__)

# 清理过期临时文件（单位：天）
environ.clean_temp_files(days=7)
```

### config — 配置管理

多层配置系统，优先级：环境变量 > 缓存 > 默认值：

```python
from linktools.core import environ

config = environ.config

# 读取配置（支持类型转换）
config.get("MY_KEY", type=int, default=0)

# 写入 / 缓存配置
config.set("MY_KEY", "value")
config.update_cache(MY_KEY="value")   # 持久化到本地缓存
config.remove_cache("MY_KEY")

# 从文件加载配置
config.update_from_file("config.yml")
```

`Config.Property` 系列描述符用于在 `configs` 字典中声明配置项，支持 `|` 运算符链式设置后备值：

```python
from linktools.core import environ, Config

environ.config.update(

    # 基础属性：直接从配置中读取，支持类型转换
    MY_KEY=Config.Property(type=str) | "default_value",

    # Alias：优先读取另一个 key，找不到时继续后备
    HTTP_PORT=Config.Alias("PORT", type=int) | Config.Prompt(cached=True) | 80,

    # Prompt：启动时交互式询问用户输入，cached=True 表示输入后缓存到本地
    ROOT_DOMAIN=Config.Alias("DOMAIN") | Config.Prompt(cached=True) | "localhost",

    # Confirm：启动时询问 yes/no，cached=True 表示缓存结果
    HTTPS_ENABLE=Config.Alias("ENABLE_HTTPS", type=bool) | Config.Confirm(cached=True) | True,

    # Lazy：根据其他配置项的值动态决定规则，lambda 接收当前 Config 对象
    HTTPS_PORT=Config.Alias("PORT", type=int) | Config.Lazy(
        lambda cfg:
        Config.Prompt(type=int, cached=True) | 443   # HTTPS 开启时询问端口
        if cfg.get("HTTPS_ENABLE")
        else Config.Property(type=int) | 0            # 未开启时固定为 0
    ),

    # Error：条件不满足时抛出错误，常与 Lazy 配合
    API_KEY=Config.Lazy(
        lambda cfg:
        Config.Error("API_KEY is required when HTTPS is enabled")
        if cfg.get("HTTPS_ENABLE")
        else Config.Property(type=str) | ""
    ),
)
```

`|` 运算符按顺序尝试每个后备值，直到某一个能成功解析为止。

### tools — 工具管理

声明式工具定义，自动处理下载、解压和执行：

```python
from linktools.core import environ

# 获取工具
tool = environ.get_tool("apktool")

# 检查是否可用
tool.exists        # 是否已下载
tool.supported     # 当前系统是否支持

# 下载并执行
tool.prepare()
tool.exec("-h")
proc = tool.popen("d", "app.apk")
```

### utils — 通用工具函数

```python
from linktools import utils

# 系统信息
utils.get_system()       # 'darwin' / 'linux' / 'windows'
utils.get_machine()      # 'x86_64' / 'arm64' / ...
utils.get_lan_ip()
utils.get_wan_ip()

# 文件操作
utils.get_file_md5("path/to/file")
utils.read_file("path/to/file")
utils.write_file("path/to/file", "content")

# 网络
utils.make_url("https://example.com", path="/api", key="val")
utils.guess_file_name("https://example.com/foo.zip")

# 进程
proc = utils.popen("adb", "devices")
utils.get_free_port()    # 获取可用端口

# 类型转换
utils.cast(int, "42", default=0)
utils.bool("true")
utils.coalesce(None, None, "fallback")  # → "fallback"
```

### decorator — 常用装饰器

```python
from linktools.decorator import cached_property, singleton, try_except

@singleton
class MyManager:
    pass

class MyClass:
    @cached_property
    def expensive(self):
        return 1 # compute()

@try_except(errors=(Exception,), default=None)
def risky():
    ...
```

### types — 公共类型与异常

```python
from linktools.types import Timeout, Stoppable, Error, ToolError

# 超时管理
timeout = Timeout(30)
timeout.remain   # 剩余秒数

# 可停止资源（实现 __stop__ 即可配合 with 使用）
class MyResource(Stoppable):
    def __stop__(self):
        ...

# 异常体系
# Error → ConfigError
#       → ToolError → ToolNotFound
#                   → ToolNotSupport
#                   → ToolExecError
#       → DownloadError
```

## 相关链接

- GitHub: <https://github.com/linktools-toolkit/linktools>
- 子包文档:
  - [linktools-common](https://github.com/linktools-toolkit/linktools/tree/master/linktools-common) — 通用工具
  - [linktools-mobile](https://github.com/linktools-toolkit/linktools/tree/master/linktools-mobile) — 移动设备工具
  - [linktools-cntr](https://github.com/linktools-toolkit/linktools/tree/master/linktools-cntr) — 容器管理工具
