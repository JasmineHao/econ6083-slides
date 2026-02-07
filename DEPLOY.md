# 部署指南 / Deployment Guide

## 📦 本地已完成 / Local Setup Complete

✅ 项目结构已创建
✅ npm 依赖已安装
✅ 幻灯片已构建到 `public/` 目录
✅ 首页已生成

## 🌐 部署到 GitHub Pages

### 步骤 1: 创建 GitHub 仓库（如果还没有）

```bash
cd D:\Dropbox\ECON6083\presentations
git init
git add .
git commit -m "Initial commit: ECON6083 presentations setup"
```

### 步骤 2: 关联远程仓库

```bash
# 在 GitHub 上创建一个新的 public 仓库（例如：econ6083-slides）
git remote add origin https://github.com/YOUR-USERNAME/econ6083-slides.git
git branch -M main
git push -u origin main
```

### 步骤 3: 配置 GitHub Pages

1. 进入仓库的 **Settings** → **Pages**
2. 在 **Source** 下选择 **"GitHub Actions"**
3. 保存设置

### 步骤 4: 触发自动部署

下次推送代码时，GitHub Actions 会自动：
- 安装依赖
- 构建所有幻灯片
- 生成首页
- 部署到 GitHub Pages

```bash
# 以后每次更新只需：
git add .
git commit -m "Update presentations"
git push
```

### 步骤 5: 访问网站

几分钟后，访问：
```
https://YOUR-USERNAME.github.io/REPO-NAME/
```

## 🔧 日常使用

### 添加新讲座

1. 在 `_slides/` 中创建新的 `.md` 文件
2. 本地预览：`npm start`
3. 构建：`npm run build && npm run build:index`
4. 提交并推送到 GitHub

### 本地预览

```bash
npm start
# 访问 http://localhost:8080
```

### 更新现有幻灯片

1. 编辑 `_slides/` 中的 `.md` 文件
2. 保存后浏览器会自动刷新（使用 `npm start` 时）
3. 提交更改并推送

## 📝 Front Matter 模板

每个新幻灯片文件开头应包含：

```yaml
---
marp: true
theme: academic
paginate: true
math: mathjax
footer: 'ECON6083 Lecture X | Topic Name'
---
```

## 🎨 自定义主题

编辑 `themes/academic.css` 来修改：
- 颜色方案
- 字体
- 布局
- 表格样式
- 代码高亮

修改后重新运行 `npm run build`。

## 📤 导出 PDF

```bash
npm run pdf
```

PDF 文件会生成在 `public/` 目录中。

## ⚠️ 注意事项

- `public/` 目录已在 `.gitignore` 中，不会提交到 Git
- GitHub Actions 会在云端重新构建，确保线上版本始终是最新的
- 图片应放在 `_slides/img/` 目录，使用相对路径引用
- MathJax 公式使用 `$...$`（行内）或 `$$...$$`（块级）

## 🆘 故障排除

### 本地构建失败
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

### GitHub Actions 失败
1. 检查 `.github/workflows/deploy.yml` 配置
2. 确认仓库的 Pages 设置正确
3. 查看 Actions 标签页的错误日志

### 样式不生效
- 确认 front matter 中 `theme: academic`
- 检查 `themes/academic.css` 文件存在
- `.marprc.yml` 中 `themeSet: ./themes/` 配置正确

---

**Created**: 2026-02-06
**Status**: ✅ Ready to deploy
