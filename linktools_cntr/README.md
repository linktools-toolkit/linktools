# Docker/Pod容器部署工具

## 开始使用

以基于debain的系统为例配置环境，其他系统请自行安装相应软件，包括Python3, Python3-pip, Git, Docker, Docker Compose

```bash
# Install Python3, Python3-pip, Git, Docker, Docker Compose
wget -qO- get.docker.com | bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip git docker-compose-plugin
```

安装最新版linktools-cntr库

```bash
python3 -m pip install -U linktools linktools-cntr
```

也可以安装开发版linktools-cntr库
```bash
python3 -m pip install --ignore-installed \
  "linktools@ git+https://github.com/linktools-toolkit/linktools.git@master" \
  "linktools_cntr@ git+https://github.com/linktools-toolkit/linktools-cntr.git@master"
```

## 容器部署

### Nas (主页、Nextcloud、...) 环境部署

👉 [搭建文档](https://github.com/ice-black-tea/cntr-homelab/blob/master/500-omv/README.md)

### Xray Server (websocket + ssl + vless) 环境搭建

👉 [搭建文档](https://github.com/ice-black-tea/cntr-homelab/blob/master/220-xray-server/README.md)

### Redroid (Redroid、Redroid-Builder) 环境搭建

👉 [搭建文档](https://github.com/redroid-rockchip)

## 内置配置

第一次部署时，会要求填写对应容器的配置项，其中部分内置的配置项包括：

| 参数                    | 类型  | 默认值                                   | 描述                                                                                                                       |
|-----------------------|-----|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| CONTAINER_TYPE        | str | -                                     | 可选项<br/>1. docker: 使用root权限docker daemon<br/>2. docker-rootless: 使用rootless模式docker daemon<br/> 3. podman: 使用podman<br/> |
| DOCKER_USER           | str | 当前shell用户                             | 部分容器rootless模式的容器以此用户权限运行                                                                                                |
| DOCKER_HOST           | str | /var/run/docker.sock                  | docker daemon进程的 url                                                                                                     |
| DOCKER_APP_PATH       | str | ~/.linktools/data/container/app       | 容器数据持久化目录，强烈建议指定到固态硬盘中                                                                                                   |
| DOCKER_APP_DATA_PATH  | str | ~/.linktools/data/container/app_data  | 不会频繁读写的容器数据持久化目录，可以放在机械硬盘中                                                                                               |
| DOCKER_USER_DATA_PATH | str | ~/.linktools/data/container/user_data | 重要用户数据目录，强烈推荐指定到nas专用硬盘中                                                                                                 |
| DOCKER_DOWNLOAD_PATH  | str | ~/.linktools/data/container/download  | 下载数据目录                                                                                                                   |
| HOST                  | str | 当前局域网ip地址                             |                                                                                                                          |

## 常用命令

```bash
# 每个子命令都可以通过添加-h参数查看帮助
ct-cntr -h

#######################
# 代码仓库相关（支持git链接和本地路径）
#######################

# 添加仓库
ct-cntr repo add https://github.com/ice-black-tea/cntr-homelab 

# 拉去仓库最新代码
ct-cntr repo update

# 删除仓库
ct-cntr repo remove

#######################
# 容器安装列表管理
#######################

# 添加容器
ct-cntr add omv gitlab portainer vscode

# 删除容器
ct-cntr remove omv

#######################
# 容器管理
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

# 查看容器docker配置
ct-cntr config

# 查看相关变量配置
ct-cntr config list

# 修改变量
ct-cntr config set ROOT_DOMAIN=test.com ACME_DNS_API=dns_ali Ali_Key=xxx Ali_Secret=yyy

# 删除变量
ct-cntr config unset ROOT_DOMAIN ACME_DNS_API Ali_Key Ali_Secret

# 使用vim编辑配置文件
ct-cntr config edit --editor vim

# 重新加载配置
ct-cntr config reload 
```
