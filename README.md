# kanosa's blog

使用 Hexo 8 和 Butterfly 5.5.2 的个人博客，站点地址为 <https://kanosa0101.github.io>。

## 本地运行

使用 Node.js 22 或更新版本，统一使用 npm 和 `package-lock.json`。

```sh
git submodule update --init --recursive
npm ci
npm run server -- --ip 127.0.0.1
```

本地访问 <http://localhost:4000>。

```sh
npm run clean
npm run build
npm run check
```

`check` 验证文章的显式发布日期、生成页面中的站内链接、本地资源、专用页面模板以及 Live2D 模型和关联资源。浏览器外部 CDN 的可用性仍需浏览器验收。

## 内容和外观

| 位置 | 用途 |
| --- | --- |
| `_config.yml` | 站点信息、文章链接、部署和 Live2D 插件 |
| `_config.butterfly.yml` | 导航、首页文案、主题组件和 CSS 注入 |
| `source/_posts/` | Markdown 文章 |
| `source/about/index.md` | 关于页面 |
| `source/link/index.md` | 友链页面，类型为 `link` |
| `source/_data/link.yml` | 日后填写真实友链时创建，格式见主题文档 |
| `source/images/` | 个人头像、背景等资源 |
| `source/css/custom.css` | 彩色标题、磨砂卡片、透明页脚、手机适配和焦点样式 |
| `themes/butterfly/` | 上游主题；尽量通过站点配置及自定义 CSS 扩展 |

新增文章：`npx hexo new "文章标题"`。在文章 front matter 中填写 `date`、`tags`、`categories` 和 `description`。

首页标题及打字机文字的颜色在 `custom.css` 的 `linear-gradient` 中调整，打字机的三句原文位于主题配置的 `subtitle.sub`。

卡片通过 `--blog-card-bg` 和 `--blog-reading-bg` 分别控制普通卡片与正文区域的透明度，支持明暗两套配色。不支持背景模糊的浏览器使用更实的底色，独立阅读模式保留主题原有行为。页脚与无背景图的内页头部使用透明底色。

## Git 和部署

源代码使用 `source` 分支，`npm run deploy` 会通过现有 Git 部署配置发布构建结果到远端 `main` 分支；它是发布操作，不是本地预览。

GitHub Actions 工作流仅构建、检查并保存 `public/` 产物，不自动发布。

主题通过 `.gitmodules` 与 gitlink 固定在上游提交 `8fedddbb5a7527c6f73311496f2fd75c166e0bf1`。站点实际使用的个人图片位于 `source/images/`，不依赖本机主题目录中未提交的图片改动；更新主题前应先保存这些本机改动。

`public/`、`db.json`、`node_modules/`、`.deploy_git/` 为生成或缓存目录，不提交。不要直接修改 `public/`。

## 后续内容

目前保留默认 Hello World 文章；关于页只使用现有昵称和公开 GitHub 地址。个人经历、正式文章、友链及评论服务需按真实信息补充。
