# Linktools Toolkit

## 开始使用

### 依赖项

python & pip (3.6及以上): <https://www.python.org/downloads/>

### 安装

使用pip安装linktools

```bash
# pip直接安装linktools，按需添加依赖项，推荐使用all添加所有依赖项
python3 -m pip install -U "linktools[all]"
# 也可以用以下命令安装github上的最新版本:
# python3 -m pip install --ignore-installed "linktools@ git+https://github.com/linktools-toolkit/linktools.git@master"
```

额外的依赖项以及相应功能可通过[requirements.yml](https://github.com/linktools-toolkit/linktools/blob/master/requirements.yml)查看

### 配置alias（推荐）

对于*nix等系统，推荐在~/.bashrc 或 ~/.bash_profile 或 ~/.zshrc等文件中配置，简化调用方式，如：

```bash
# 对于未正确设置PATH环境变量，或者使用venv安装模块，会出现命令找不到的情况（command not found: ct-env）
# 可通过以下命令生成alias脚本添加相关命令
# 需要注意此处python3需要替换成自己安装环境下的interpreter，比如~/projects/linktools/venv/bin/python
# ps. 如果已安装argcomplete库，还可以生成自动补全脚本
eval "$(python3 -m linktools.cli.env alias --shell bash)"

# 配置全局java环境，指定java版本号（如：11.0.23/17.0.11/21.0.3）
# 可通过 https://sap.github.io/SapMachine/#download 查找LTS版本号
eval "$(ct-env --silent java 17.0.11 --shell bash)"

# alias简化调用
alias adb="at-adb"
alias sib="it-sib"
alias pidcat="at-pidcat"

# alias简化各类工具调用
alias apktool="ct-tools apktool"
alias burpsuite="ct-tools burpsuite"
alias jadx="ct-tools --set version=1.5.0 jadx-gui"  # 指定jadx版本号
```

## 相关功能

```
$ python3 -m linktools
    ___       __   __              __
   / (_)___  / /__/ /_____  ____  / /____
  / / / __ \/ //_/ __/ __ \/ __ \/ / ___/  linktools toolkit (v0.0.1.dev0)
 / / / / / / ,< / /_/ /_/ / /_/ / (__  )   by: Hu Ji <669898595@qq.com>
/_/_/_/ /_/_/|_|\__/\____/\____/_/____/

📎 All commands
├── 📖 at: Android scripts
│   ├── 👉 adb: Manage multiple Android devices effortlessly with adb commands
│   ├── 👉 agent: Debug and interact with android-tools.apk for troubleshooting
│   ├── 👉 app: Retrieve detailed information about installed applications on Android devices
│   ├── 📘 cert: Display detailed X.509 certificate information for secure communication
│   ├── 👉 debug: Debug Android apps effectively using the Java Debugger (jdb)
│   ├── 👉 frida: Use Frida for dynamic analysis on rooted Android devices
│   ├── 👉 info: Collect detailed device information
│   ├── 📘 intent: Execute common Android intent actions for automation and testing
│   ├── 👉 objection: Simplify security testing with Objection on rooted Android devices
│   ├── 👉 pidcat: Filter logcat by package name
│   └── 👉 top: Fetch basic information about the currently running application
├── 📖 ct: Common scripts
│   ├── 📘 env: Manage and configure the Linktools environment
│   ├── 👉 grep: Search and match files using regular expressions
│   └── 👉 tools: Execute tools directly from remote URLs
└── 📖 it: iOS scripts
    ├── 👉 frida: Use Frida for dynamic analysis on jailbroken iOS devices
    ├── 👉 ipa: Parse and extract detailed information from IPA files
    ├── 👉 objection: Simplify security testing with Objection on jailbroken devices
    ├── 👉 scp: Securely copy files to/from a jailbroken iOS device using OpenSSH
    ├── 👉 sib: Manage multiple iOS devices effortlessly with sib commands
    └── 👉 ssh: Remotely login to jailbroken iOS devices using the OpenSSH client
```

### 通用功能（脚本前缀为ct-）

#### 👉 ct-env

<details>
<summary>环境配置相关命令</summary>

##### 常用命令

```bash
# 生成alias脚本，常配合~/.bashrc等文件使用
$ ct-env --silent alias --shell bash

# 生成自动补全脚本，常配合~/.bashrc等文件使用
$ ct-env --silent completion --shell bash

# 生成配置java环境变量脚本，常配合~/.bashrc等文件使用
$ ct-env --silent java 17.0.11 --shell bash

# 进入已初始化相关环境变量的shell
$ ct-env shell

# 清除项目中7天以上未使用的缓存文件
$ ct-env clean 7
```

</details>

#### 👉 ct-grep

<details>
<summary>类似linux中的grep，正则匹配文件内容 ，额外添加解析zip、elf等格等功能</summary>

![ct-grep](https://raw.githubusercontent.com/linktools-toolkit/linktools/master/images/ct-grep.png)

</details>

#### 👉 ct-tools

<details>
<summary>读取配置文件，即可下载使用对应工具，声明了adb、jadx、apktool、baksmali等常用工具</summary>

##### 常用命令

所有声明的工具可通过[配置文件](https://github.com/linktools-toolkit/linktools/blob/master/src/linktools/template/tools.yml)查看，此处以apktool举例

```bash
# 初始化并执行apktool命令
$ ct-tools apktool -h

# 查看apktool相关配置
$ ct-tools --config apktool

# 只初始化不执行
$ ct-tools --download apktool

# 清除apktool相关文件
$ ct-tools --clear apktool

# 后台运行apktool
$ ct-tools --daemon apktool

# 修改apktool版本号
$ ct-tools --set version=2.5.0 apktool
```

</details>

### android相关功能（脚本前缀为at-）

#### 👉 at-adb

<details>
<summary>若环境变量中存在adb，则直接执行，否则自动下载最新版本。该功能支持操作多台手机</summary>

##### 常用命令

at-adb的命令与adb命令一致，以下以adb shell举例

```bash
# 指定序列号，并调用adb shell
$ at-adb -s xxx shell

# 上次使用的设备，并调用adb shell
$ at-adb -l shell

# 连接远程端口，并调用adb shell
$ at-adb -c 127.0.0.1:5555 shell

# 未指定则会需要选择一台设备，并调用adb shell
$ at-adb shell
More than one device/emulator
>> 1: 18201FDF6003BE (Pixel 6)
   2: 10.10.10.58:5555 (Pixel 6)
Choose device [1~2] (1): 1
```

</details>

#### 👉 at-pidcat

<details>
<summary>集成了pidcat，并且修复了中文字符宽度问题，原项目链接：https://github.com/JakeWharton/pidcat</summary>

##### 常用命令

```bash
# 查看指定包名应用的日志
$ at-pidcat -p me.ele

# 查看当前运行进程的日志
$ at-pidcat --top

# 查看指定tag的日志
$ at-pidcat -t XcdnEngine
```

</details>

#### 👉 at-top

<details>
<summary>显示顶层应用信息、获取顶层应用apk、截屏等</summary>

##### 常用命令

```bash
# 展示当前顶层应用包名、activity、apk路径等信息
$ at-top 

# 将当前顶层应用apk导出
$ at-top --apk

# 将当前页面截屏导出
$ at-top --screen
```

</details>

#### 👉 at-app

<details>
<summary>通过执行agent调用pms读取app基本信息并展示，组件、权限等信息相对静态检测更为准确</summary>

##### 常用命令

```bash
# 显示当前应用的基本信息
$ at-app

# 显示当前应用的详细信息
$ at-app --detail

# 显示当前应用信息风险项
$ at-app --detail --dangerous

# 显示非系统应用信息
$ at-app --non-system
```

##### 输出效果

![at-app](https://raw.githubusercontent.com/linktools-toolkit/linktools/master/images/at-app.png)

</details>

#### 👉 at-inetnt

<details>
<summary>打包了常用intent操作，支持如打开设置界面、开发者选项界面、app设置界面、安装证书、打开浏览器链接等功能</summary>

##### 常用命令

```bash
# 跳转到设置页
$ at-intent setting

# 跳转到开发者选项页
$ at-intent setting-dev

# 跳转到app设置页
$ at-intent setting-app

# 安装证书
$ at-intent setting-cert ~/test.crt

# 安装apk
$ at-intent install https://example.com/test.apk

# 浏览器中打开特定页，也可用于测试url scheme
$ at-intent browser https://example.com
```

</details>

#### 👉 at-frida

<details>
<summary>该功能旨在方便使用frida，可自动下载server，支持加载远程脚本，并内置了常用功能</summary>

##### 相关特性
1. 可以支持根据android设备和python的frida版本，全自动完成下载、推送、运行frida server
2. 监听了spawn进程变化情况，可以同时hook主进程和各个子进程
3. 监听js文件变化，实时加载
4. 注入了内置脚本，封装常用功能，如：过ssl pinning
5. 支持加载远程脚本
6. 支持重定向设备流量到本地端口

##### 使用方式

**1) 以命令行方式运行**

```bash
# 从本地加载~/test/frida.js脚本，以spawn模式注入到me.ele进程中
$ at-frida -l ~/test/frida.js -p me.ele --spawn

# 从远程加载frida脚本，注入到me.ele进程中，并将me.ele流量重定向到本地8080端口
$ at-frida -c https://raw.githubusercontent.com/linktools-toolkit/linktools/master/agents/frida/test/android.js -p me.ele --redirect-port 8080

# 只启动frida-server，不注入脚本
$ at-frida --serve --remote-port 27042 --local-port 27042 -p fake_package

# 不启动frida-server，通过设备上frida server启动的27042端口，注入到me.ele进程中
$ at-frida --no-serve --remote-port 27042 -p me.ele
```

**2) 使用python方式调用**

执行如下python脚本即可自动开启frida-server，并将js代码注入到指定进程，参考[src/linktools/cli/commands/android/frida.py](https://github.com/linktools-toolkit/linktools/blob/master/src/linktools/cli/commands/android/frida.py) 

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.cli import BaseCommand
from linktools.frida import FridaApplication, FridaEvalCode, FridaAndroidServer


class Command(BaseCommand):

    def init_arguments(self, parser):
        pass
    
    def run(self, args):
        code = """
            Java.perform(function () {
                JavaHelper.hookMethods(
                    "java.util.HashMap",
                     "put",
                     {stack: false, args: true}
                );
            });
            """
        
        with FridaAndroidServer() as server:
            app = FridaApplication(
                server,
                user_scripts=(FridaEvalCode(code),),
                enable_spawn_gating=True,
                target_identifiers=rf"^com.topjohnwu.magisk($|:)"
            )
            app.inject_all()
            app.run()


command = Command()
if __name__ == "__main__":
    command.main()
```

##### 内置接口

e.g. [java相关接口](https://github.com/linktools-toolkit/linktools/blob/master/agents/frida/lib/java.ts)

```javascript
Java.perform(function () {

    // hook特定类的指定方法
    JavaHelper.hookMethod(
        "me.ele.privacycheck.f",                    // 可以是类名，也可以是类对象 => Java.use("me.ele.privacycheck.f")
        "a",                                        // 方法名
        ['android.app.Application', 'boolean'],     // 参数类型
        function (obj, args) {                      // hook方法实现
            args[1] = true;
            return this(obj, args);                 // this代表当前hook方法，obj代表当前hook对象，args代表当前hook方法参数
        }
    );

    // hook特定类的所有名为isHttpType的方法
    JavaHelper.hookMethods(
        "anet.channel.entity.ConnType",             // 可以是类名，也可以是类对象
        "isHttpType",                               // 方法名
        () => true                                  // hook实现
    );
    
    // hook特定类的所有方法
    JavaHelper.hookAllMethods(
        "p.r.o.x.y.PrivacyApi",                     // 可以是类名，也可以是类对象
        JavaHelper.getEventImpl({                   // 生成一个通用的hook方法
            stack: true,                            // 打印堆栈
            args: true,                             // 打印参数返回值
            thread: false,
            extras: {                               // 自定义参数，会回显日志中
                customKey1: "自定义参数",
            }
        })
    );
    
    // 类似Java.use()
    // 如果当前classloader不存在需要找的类，则会持续监控动态加载的classloader，直到找到指定类为止
    JavaHelper.use("p.r.o.x.y.PrivacyApi", function(clazz) {
        // 终于等到class出现，干点想干的事吧
        JavaHelper.hookAllMethods(
            clazz,
            JavaHelper.getEventImpl({
                stack: true,
                args: true,
            })
        );
    });
    
    // 禁用ssl pinning
    JavaHelper.bypassSslPinning();
    
    // 开启webview调试模式
    JavaHelper.setWebviewDebuggingEnabled();
    
});
```

</details>

#### 👉 at-agent

<details>
<summary>测试android-tools.apk时使用</summary>

##### 常用命令

```bash
# 调用android-tools.apk中的方法
$ at-agent common --set-clipboard "剪切板内容"

# 获取剪切板内容
$ at-agent common --get-clipboard

# 以root权限dump系统服务信息，包括服务所在进程信息，需要root设备并且挂载DebugFS：https://source.android.com/docs/core/architecture/kernel/using-debugfs-12?hl=zh-cn
$ at-agent -u root --debug service --detail

# 添加插件并调用插件方法
$ at-agent --plugin app-release.apk
```

</details>

### ios相关功能（脚本前缀为it-）

#### 👉 it-frida

<details>
<summary>该功能旨在方便使用frida，支持加载远程脚本，内置了常用功能</summary>

```
$ it-frida -h                                                                                                                                       ░▒▓ ✔  12:37:52
usage: it-frida [-h] [--version] [--verbose] [--debug] [--time | --no-time] [--level | --no-level] [-u UDID | --connect IP:PORT | --last] [-b BUNDLE_ID] [--spawn]
                [-P KEY VALUE] [-l SCRIPT] [-e CODE] [-c URL] [-a]

Easy to use frida (require iOS device jailbreak)

    ___       __   __              __
   / (_)___  / /__/ /_____  ____  / /____
  / / / __ \/ //_/ __/ __ \/ __ \/ / ___/  linktools toolkit (v0.0.1.dev0)
 / / / / / / ,< / /_/ /_/ / /_/ / (__  )   by: Hu Ji <669898595@qq.com>
/_/_/_/ /_/_/|_|\__/\____/\____/_/____/

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -b BUNDLE_ID, --bundle-id BUNDLE_ID
                        target bundle id (default: frontmost application)
  --spawn               inject after spawn (default: false)
  -P KEY VALUE, --parameters KEY VALUE
                        user script parameters
  -l SCRIPT, --load SCRIPT
                        load user script
  -e CODE, --eval CODE  evaluate code
  -c URL, --codeshare URL
                        load share script url
  -a, --auto-start      automatically start when all processes exits

log arguments:
  --verbose             increase log verbosity
  --debug               enable debug mode and increase log verbosity
  --time, --no-time     show log time
  --level, --no-level   show log level

sib arguments:
  -u UDID, --udid UDID  specify unique device identifier
  --connect IP:PORT     use device with TCP/IP
  --last                use last device
```

</details>
