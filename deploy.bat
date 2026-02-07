@echo off
chcp 65001 > nul
echo ========================================
echo ECON6083 课件网站一键部署脚本
echo ========================================
echo.

REM 步骤 1: 检查是否已添加远程仓库
echo [1/5] 检查远程仓库配置...
git remote -v > nul 2>&1
if %errorlevel% equ 0 (
    git remote get-url origin > nul 2>&1
    if %errorlevel% equ 0 (
        echo ✓ 远程仓库已配置
        set REMOTE_EXISTS=1
    ) else (
        set REMOTE_EXISTS=0
    )
) else (
    set REMOTE_EXISTS=0
)

if %REMOTE_EXISTS% equ 0 (
    echo.
    echo ⚠ 尚未配置远程仓库
    echo.
    echo 请先在 GitHub 上创建一个新仓库：
    echo 1. 访问 https://github.com/new
    echo 2. 仓库名称输入: econ6083-slides
    echo 3. 设置为 Public（公开）
    echo 4. 不要勾选 "Add a README file"
    echo 5. 点击 "Create repository"
    echo.
    set /p GITHUB_USERNAME="请输入你的 GitHub 用户名: "
    echo.
    echo 正在添加远程仓库...
    git remote add origin https://github.com/!GITHUB_USERNAME!/econ6083-slides.git
    if %errorlevel% neq 0 (
        echo ✗ 添加远程仓库失败
        pause
        exit /b 1
    )
    echo ✓ 远程仓库已添加
)

REM 步骤 2: 检查是否已提交
echo.
echo [2/5] 检查本地提交状态...
git log -1 > nul 2>&1
if %errorlevel% neq 0 (
    echo 正在添加文件到 Git...
    git add .
    git commit -m "Initial commit: ECON6083 presentations setup

- Marp-based presentation system
- Academic theme with gradient design
- GitHub Actions auto-deployment
- First lecture: DAGs and Causal Identification

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
    if %errorlevel% neq 0 (
        echo ✗ 提交失败
        pause
        exit /b 1
    )
    echo ✓ 文件已提交
) else (
    echo ✓ 已有提交记录
    REM 检查是否有未提交的更改
    git status --porcelain | findstr /r "." > nul
    if %errorlevel% equ 0 (
        echo 发现未提交的更改，正在提交...
        git add .
        git commit -m "Update presentations"
        echo ✓ 更改已提交
    )
)

REM 步骤 3: 设置主分支名称
echo.
echo [3/5] 设置主分支...
git branch -M main
echo ✓ 主分支已设置为 main

REM 步骤 4: 推送到 GitHub
echo.
echo [4/5] 推送到 GitHub...
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo ⚠ 推送失败。这可能是因为：
    echo 1. 需要 GitHub 身份验证
    echo 2. 远程仓库还未创建
    echo 3. 网络连接问题
    echo.
    echo 请按照以下步骤手动操作：
    echo.
    echo 如果你还没有创建 GitHub 仓库：
    echo 1. 访问 https://github.com/new
    echo 2. 仓库名: econ6083-slides
    echo 3. 设置为 Public
    echo 4. 不要添加 README
    echo 5. 创建后运行: git push -u origin main
    echo.
    echo 如果你需要身份验证：
    echo 1. 访问 https://github.com/settings/tokens
    echo 2. 生成一个新的 Personal Access Token
    echo 3. 使用 token 作为密码重新推送
    echo.
    pause
    exit /b 1
)
echo ✓ 代码已推送到 GitHub

REM 步骤 5: 配置提示
echo.
echo [5/5] GitHub Pages 配置
echo.
echo ✓ 代码已成功推送！
echo.
echo 📝 最后一步：配置 GitHub Pages
echo.
echo 请访问你的仓库设置页面并完成以下配置：
for /f "delims=" %%i in ('git remote get-url origin') do set REPO_URL=%%i
set REPO_URL=%REPO_URL:.git=%
echo.
echo 1. 打开: %REPO_URL%/settings/pages
echo 2. 在 "Source" 下拉菜单中选择 "GitHub Actions"
echo 3. 保存设置
echo.
echo ⏳ 等待 2-3 分钟让 GitHub Actions 完成构建
echo.
echo 🎉 完成后，你的网站将发布在：
for /f "tokens=3 delims=/" %%a in ("%REPO_URL%") do set USERNAME=%%a
for /f "tokens=4 delims=/" %%a in ("%REPO_URL%") do set REPONAME=%%a
echo https://%USERNAME%.github.io/%REPONAME%/
echo.
echo ========================================
echo 🎓 后续更新课件
echo ========================================
echo.
echo 1. 在 _slides/ 中编辑或添加 .md 文件
echo 2. 运行: npm run build
echo 3. 运行: npm run build:index
echo 4. 提交并推送:
echo    git add .
echo    git commit -m "Update slides"
echo    git push
echo.
echo 💡 提示：每次 git push 后，GitHub Actions 会自动重新部署网站
echo.
pause
