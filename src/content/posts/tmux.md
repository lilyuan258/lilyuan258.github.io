---
title: tmux 入门
description: >-
  这篇笔记记录本机安装 gpakosz/.tmux 后的实际用法。 tmux 的核心作用是把终端会话留在后台：即使 SSH
  断开、终端窗口关闭，训练、服务、日志监控等任务也可以继续运行；重新连接后还能回到同一个现场。
pubDate: '2026-07-09'
updatedDate: '2026-07-09'
tags:
  - linux
  - terminal
  - tmux
  - server
  - dexrobot
draft: false
source: tmux入门.md
wordCount: 1347
readingTime: 3
---
这篇笔记记录本机安装 `gpakosz/.tmux` 后的实际用法。`tmux` 的核心作用是把终端会话留在后台：即使 SSH 断开、终端窗口关闭，训练、服务、日志监控等任务也可以继续运行；重新连接后还能回到同一个现场。

## 本机安装状态

本次安装的是这个配置：

```text
https://github.com/gpakosz/.tmux
```

当前状态：

| 项目 | 路径或状态 |
|---|---|
| tmux 版本 | `tmux 3.4` |
| 配置仓库 | `/home/lcy/.tmux` |
| 主配置链接 | `/home/lcy/.tmux.conf -> .tmux/.tmux.conf` |
| 本地配置文件 | `/home/lcy/.tmux.conf.local` |
| 原配置备份 | `/home/lcy/.tmux-config-backups/.tmux.conf.20260709-135756` |
| 保留的旧设置 | `set -g extended-keys on` |

原来的 `~/.tmux.conf` 只有一行：

```tmux
set -g extended-keys on
```

安装后这行已经保留到 `/home/lcy/.tmux.conf.local` 的 local overrides 区域。

## 快捷键记法

tmux 文档里经常用缩写表示组合键：

| 写法 | 含义 |
|---|---|
| `C-h` | `Ctrl-h`，也就是按住 `Ctrl` 再按 `h` |
| `C-l` | `Ctrl-l` |
| `M-u` | `Alt-u`，`M` 通常表示 Meta，在多数键盘上就是 `Alt` |
| `<prefix>` | tmux 前缀键，先按它，再按后面的命令键 |

因此：

```text
<prefix> C-h
```

意思是：先按 tmux prefix，松开，然后按 `Ctrl-h`。

在这份配置里，prefix 有两个都能用：

| Prefix | 说明 |
|---|---|
| `Ctrl-b` | tmux 默认 prefix |
| `Ctrl-a` | `gpakosz/.tmux` 增加的常用 prefix |

例如：

```text
<prefix> c
```

可以实际按成：

```text
Ctrl-a c
```

或者：

```text
Ctrl-b c
```

如果在 shell 里想输入真正的 `Ctrl-a`，例如跳到当前命令行开头，可以按：

```text
Ctrl-a Ctrl-a
```

## 核心概念

`tmux` 里常用三个层级：

| 概念 | 说明 |
|---|---|
| `session` | 会话，一个长期存在的工作空间 |
| `window` | 窗口，类似浏览器标签页 |
| `pane` | 分屏，一个窗口里可以切成多个终端区域 |

可以这样理解：

```text
tmux
└── session: vla
    ├── window 0: main
    │   ├── pane 0: 编辑和运行命令
    │   └── pane 1: 查看日志或 GPU
    └── window 1: tests
        └── pane 0: 跑测试或监控任务
```

## 常用命令

新建一个会话：

```bash
tmux new -s work
```

进入已有会话：

```bash
tmux attach -t work
```

简写：

```bash
tmux a -t work
```

查看所有会话：

```bash
tmux ls
```

退出当前 tmux 会话但不关闭任务：

```text
<prefix> d
```

关闭指定会话：

```bash
tmux kill-session -t work
```

谨慎使用：

```bash
tmux kill-server
```

`kill-server` 会关闭所有 tmux 会话。

## 这份配置最常用的快捷键

窗口类似标签页，pane 类似分屏。

| 快捷键 | 作用 |
|---|---|
| `<prefix> d` | detach，离开 tmux，但里面的任务继续运行 |
| `<prefix> c` | 新建 window |
| `<prefix> C-h` | 切到上一个 window，也就是先按 prefix，再按 `Ctrl-h` |
| `<prefix> C-l` | 切到下一个 window，也就是先按 prefix，再按 `Ctrl-l` |
| `<prefix> Tab` | 回到上一个活跃 window |
| `<prefix> w` | 打开 window 列表 |
| `<prefix> s` | 打开 session 列表 |
| `<prefix> -` | 上下分屏 |
| `<prefix> _` | 左右分屏 |
| `<prefix> h/j/k/l` | 在 pane 之间移动 |
| `<prefix> H/J/K/L` | 调整 pane 大小 |
| `<prefix> +` | 把当前 pane 临时最大化到新 window，再按一次恢复 |
| `<prefix> z` | tmux 原生 zoom 当前 pane，再按一次恢复 |
| `<prefix> x` | 关闭当前 pane |
| `<prefix> &` | 关闭当前 window |
| `<prefix> r` | 重新加载配置 |
| `<prefix> e` | 打开 `/home/lcy/.tmux.conf.local` 进行编辑 |

这份配置里，新分出来的 pane 会保留当前目录。因此在项目目录里分屏后，新 pane 仍然会在同一个项目目录，适合开发、训练和看日志。

## 鼠标用法

这份配置增强了鼠标支持：

- 滚轮可以查看历史输出。
- 拖动 pane 边框可以调整大小。
- 点击状态栏可以切换 window。
- 右键 pane 会出现菜单，可以 split、kill、zoom、copy line 等。

鼠标模式开关：

```text
<prefix> m
```

状态栏右侧出现鼠标标记时，说明鼠标模式开启。

## 复制、滚动和搜索

进入复制模式：

```text
<prefix> Enter
```

也可以用 tmux 默认方式：

```text
<prefix> [
```

复制模式基本按 Vim 习惯操作：

| 按键 | 作用 |
|---|---|
| `h/j/k/l` | 移动 |
| `/` | 向下搜索 |
| `?` | 向上搜索 |
| `n` | 下一个搜索结果 |
| `N` | 上一个搜索结果 |
| `Space` | 开始选择 |
| `Enter` | 复制并退出 |
| `q` | 退出复制模式 |

粘贴 tmux buffer：

```text
<prefix> p
```

选择历史 buffer 粘贴：

```text
<prefix> P
```

当前 `/home/lcy/.tmux.conf.local` 里有这一项：

```tmux
tmux_conf_copy_to_os_clipboard=false
```

因此复制内容默认进 tmux 自己的 buffer，不一定同步到系统剪贴板。如果希望 tmux 内复制自动进入系统剪贴板，可以改成：

```tmux
tmux_conf_copy_to_os_clipboard=true
```

然后重新加载配置：

```text
<prefix> r
```

## 修改配置

不要直接修改：

```text
/home/lcy/.tmux/.tmux.conf
```

这个文件来自上游仓库，后续更新时可能变化。

自己的配置写在：

```text
/home/lcy/.tmux.conf.local
```

快捷打开本地配置：

```text
<prefix> e
```

修改后重新加载：

```text
<prefix> r
```

或者在命令行执行：

```bash
tmux source-file ~/.tmux.conf
```

## 插件管理

这份配置内置了 TPM 管理逻辑。插件配置写在 `/home/lcy/.tmux.conf.local` 的 plugin 区域，例如：

```tmux
set -g @plugin 'tmux-plugins/tmux-resurrect'
```

常用插件快捷键：

| 快捷键 | 作用 |
|---|---|
| `<prefix> I` | 安装插件 |
| `<prefix> u` | 更新插件 |
| `<prefix> M-u` | 卸载不再配置的插件，也就是先按 prefix，再按 `Alt-u` |

不要额外添加下面这两行，因为 `gpakosz/.tmux` 已经处理了 TPM：

```tmux
set -g @plugin 'tmux-plugins/tpm'
run '~/.tmux/plugins/tpm/tpm'
```

## DexRobot 项目推荐工作流

进入项目目录：

```bash
cd ~/dexrobot-intern/vla_pipeline
```

创建项目会话：

```bash
tmux new -s vla
```

常见窗口和分屏安排：

| 区域 | 用途 |
|---|---|
| window 0 / pane 0 | 编辑、运行普通命令 |
| window 0 / pane 1 | `watch -n 1 nvidia-smi` 或查看日志 |
| window 1 | 运行训练、服务、测试等长任务 |
| window 2 | git、实验记录、临时命令 |

示例：

```text
<prefix> -      分一个 pane 跑训练或服务
<prefix> _      再分一个 pane 看日志或 GPU
<prefix> c      新建 window 做 git、测试、编辑
<prefix> d      离开 tmux，让任务继续跑
```

下次回到现场：

```bash
tmux attach -t vla
```

## 最小必背快捷键

刚开始只需要熟练这些：

| 快捷键 | 作用 |
|---|---|
| `Ctrl-a d` | 离开 tmux，会话继续运行 |
| `Ctrl-a c` | 新建窗口 |
| `Ctrl-a -` | 上下分屏 |
| `Ctrl-a _` | 左右分屏 |
| `Ctrl-a h/j/k/l` | 切换分屏 |
| `Ctrl-a Enter` | 进入复制/滚动模式 |
| `Ctrl-a r` | 重新加载配置 |

掌握这几个就能覆盖大多数服务器开发、训练和调试场景。
