# Lilyuan 个人博客

这是一个基于 Astro 的个人主页和 Markdown 博客，用来发布 Obsidian 中关于具身智能、VLA、π 系列模型和机器人学习的笔记。

## 目录说明

```text
blog/
├─ src/content/posts/      # 同步后生成的公开 Markdown 副本
├─ public/notes-assets/    # 同步后复制的公开附件
├─ scripts/sync-notes.mjs  # 从 Obsidian 同步笔记的脚本
├─ src/pages/              # 首页、笔记索引页、文章页
└─ .github/workflows/      # GitHub Pages 自动部署配置
```

## 本地使用

第一次使用：

```powershell
cd blog
npm install
npm run sync
npm run dev
```

之后每次在 Obsidian 里写完新笔记，发布前运行：

```powershell
cd blog
npm run sync
npm run build
```

`npm run sync` 默认会读取上一级目录的 Markdown 笔记，也就是：

```text
F:\Obisidian笔记库\Atlas\Notes\具身智能paper学习
```

附件默认会从这里查找：

```text
F:\Obisidian笔记库\x\Attachments
```

同步脚本会转换 Obsidian 的 `[[双链]]`、`![[图片]]`，并把可公开附件复制到：

```text
blog/public/notes-assets/
```

## 自定义同步路径

如果以后移动了博客目录，可以用环境变量指定 Obsidian 笔记和附件路径：

```powershell
$env:OBSIDIAN_SOURCE_DIR="F:\Obisidian笔记库\Atlas\Notes\具身智能paper学习"
$env:OBSIDIAN_ATTACHMENT_DIRS="F:\Obisidian笔记库\x\Attachments"
npm run sync
```

多个附件目录可以用英文分号分隔：

```powershell
$env:OBSIDIAN_ATTACHMENT_DIRS="F:\Obisidian笔记库\x\Attachments;F:\其他附件目录"
```

## 公网访问

本地的 `npm run dev` 或 `npm run preview` 只能在你自己的电脑上访问。要让别人通过公网访问，推荐部署到 GitHub Pages。

推荐方式：

1. 在 GitHub 创建公开仓库：`lilyuan258.github.io`
2. 把 `blog` 目录里的内容作为仓库根目录提交上去
3. 在 GitHub 仓库的 `Settings -> Pages` 中，将 Source 设为 `GitHub Actions`
4. 本地更新笔记后执行：

```powershell
cd blog
npm run sync
npm run build
git add .
git commit -m "update notes"
git push
```

推送到 GitHub 后，`.github/workflows/deploy.yml` 会自动构建并发布。发布成功后，公网地址通常是：

```text
https://lilyuan258.github.io/
```

如果你不是使用 `lilyuan258.github.io` 这个个人主页仓库，而是使用项目仓库，例如 `my-blog`，需要设置：

```text
SITE_URL=https://lilyuan258.github.io
SITE_BASE=/my-blog
```

## 常用命令

```powershell
npm run sync       # 从 Obsidian 生成公开内容
npm run dev        # 本地开发预览
npm run build      # 类型检查并生成静态站点
npm run preview    # 预览构建后的 dist 目录
npm run check:site # Playwright 站点检查
```
