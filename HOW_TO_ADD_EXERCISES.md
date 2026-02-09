# 📝 如何添加练习和作业

快速指南：向网站添加新的练习或作业

---

## 🎯 添加课堂练习 (In-Class Exercise)

### 步骤 1: 复制练习文件

```bash
# 从课件目录复制到 presentations/exercises/
cp 课件/LectureXX-Problem.md presentations/exercises/
```

### 步骤 2: 配置到对应讲次

编辑 `presentations/scripts/build-index.js`，找到 `lectureResources` 配置：

```javascript
const lectureResources = {
  // ... 其他讲次

  5: {  // 要添加练习的讲次编号
    inClassExercise: {
      title: 'DML Problem',                      // 显示的标题
      file: 'Lecture05-DML-Problem.md',          // 文件名
      type: 'exercise'
    },
    assignments: []
  },

  // ... 更多讲次
};
```

### 步骤 3: 测试和部署

```bash
cd presentations

# 本地测试
npm run build:index

# 检查生成的 index.html
# 应该在 Lecture 5 的右侧看到新的练习

# 提交并推送
git add exercises/ scripts/build-index.js
git commit -m "Add Lecture 5 in-class exercise"
git push origin main
```

---

## 📚 添加作业 (Assignment)

### 步骤 1: 创建作业文件夹

```bash
cd presentations/assignments

# 创建新作业文件夹
mkdir A2

# 复制作业文件
cp ../../作业/A2/student-template/README.md A2/
cp ../../作业/A2/student-template/*.py A2/
cp ../../作业/A2/student-template/*.md A2/
```

### 步骤 2: 在 build-index.js 中注册作业

**2a. 添加作业信息：**

```javascript
const assignmentInfo = {
  'A0': { /* ... */ },
  'A1': { /* ... */ },

  'A2': {  // 新作业
    title: 'A2: Classification & Trees',
    description: 'Decision trees and random forests',
    due: 'Week 6',
    weight: '10%'
  }
};
```

**2b. 关联到讲次：**

```javascript
const lectureResources = {
  // ...

  3: {  // 在 Lecture 3 发布 A2
    inClassExercise: null,  // 如果没有课堂练习
    assignments: ['A2']      // 添加到这里
  },

  // ...
};
```

### 步骤 3: 部署

```bash
cd presentations
npm run build:index

git add assignments/ scripts/build-index.js
git commit -m "Add Assignment A2"
git push origin main
```

---

## 🔄 完整示例：添加 Lecture 6 的资源

假设我们要为 Lecture 6 添加：
- 课堂练习: HTE Problem
- 作业发布: A3

### 1. 复制文件

```bash
# 练习
cp 课件/Lecture06-HTE-Problem.md presentations/exercises/

# 作业
mkdir presentations/assignments/A3
cp 作业/A3/student-template/* presentations/assignments/A3/
```

### 2. 编辑 build-index.js

```javascript
// 添加作业信息
const assignmentInfo = {
  // ... 已有的 A0, A1, A2
  'A3': {
    title: 'A3: Heterogeneous Treatment Effects',
    description: 'Causal forests and CATE estimation',
    due: 'Week 8',
    weight: '15%'
  }
};

// 配置 Lecture 6
const lectureResources = {
  // ... 前面的讲次

  6: {
    inClassExercise: {
      title: 'HTE Problem',
      file: 'Lecture06-HTE-Problem.md',
      type: 'exercise'
    },
    assignments: ['A3']
  },

  // ... 后面的讲次
};
```

### 3. 构建和部署

```bash
cd presentations
npm run build:index

# 检查本地 public/index.html
# 确认 Lecture 6 显示正确

git add -A
git commit -m "Add Lecture 6 exercise and Assignment 3"
git push origin main

# 等待 2-3 分钟，网站自动更新
```

---

## 🎨 高级：自定义资源卡片

如果你想添加其他类型的资源（如：阅读材料、视频），可以修改 `build-index.js`：

### 添加新的资源类型

```javascript
const lectureResources = {
  7: {
    inClassExercise: { /* ... */ },
    assignments: ['A3'],

    // 新增：阅读材料
    readings: [
      {
        title: 'Pearl (2009) Chapter 3',
        url: 'https://example.com/pearl-ch3.pdf',
        type: 'pdf'
      }
    ],

    // 新增：视频
    videos: [
      {
        title: 'Lecture Recording',
        url: 'https://youtube.com/watch?v=...',
        duration: '1h 30m'
      }
    ]
  }
};
```

然后在 HTML 生成部分添加对应的卡片生成代码。

---

## 📋 快速检查清单

添加新资源前检查：

- [ ] 文件已复制到正确位置
  - 练习 → `presentations/exercises/`
  - 作业 → `presentations/assignments/AX/`

- [ ] `build-index.js` 已更新
  - 作业信息在 `assignmentInfo`
  - 讲次关联在 `lectureResources`

- [ ] 本地测试通过
  - `npm run build:index` 无错误
  - 检查生成的 `public/index.html`

- [ ] Git 提交完整
  - 添加了所有新文件
  - 提交信息清晰
  - 已推送到 GitHub

- [ ] 网站已更新
  - 等待 2-3 分钟
  - 访问网站强制刷新

---

## 🆘 常见问题

### Q: 练习文件链接打不开？
A: 检查 `file` 字段是否与实际文件名完全匹配（包括大小写）

### Q: 作业不显示？
A: 确保：
1. 作业 key (如 'A2') 在 `assignmentInfo` 中定义
2. 在 `lectureResources` 中正确引用（字符串完全匹配）

### Q: 本地测试正常，但网站没更新？
A:
1. 检查 GitHub Actions 是否成功运行
2. 等待 3-5 分钟（有时需要更长时间）
3. 清除浏览器缓存（Ctrl+F5）

### Q: 想修改卡片样式？
A: 编辑 `build-index.js` 中的 CSS 部分（`<style>` 标签内）

---

## 📖 参考

- 现有配置：`presentations/scripts/build-index.js`
- 网站设计文档：`presentations/WEBSITE_REDESIGN.md`
- 讲次顺序：`presentations/LECTURE_SEQUENCE.md`

---

**祝顺利添加新资源！** 🎓
