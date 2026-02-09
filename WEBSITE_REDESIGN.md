# ✅ 网站重新设计完成

**更新时间:** 2026-02-09
**重大更新:** 新的两栏布局，添加练习和作业

---

## 🎨 新设计特点

### 布局改进
```
旧版本：单栏列表
Lecture 1 → 链接
Lecture 2 → 链接
...

新版本：左右两栏
┌─────────────────────────────────────────────────┐
│  01 │ Slides                │ Exercises          │
│     │ View Slides →         │ • In-Class Problem │
│     │                       │ • Assignment A1    │
├─────────────────────────────────────────────────┤
│  02 │ Slides                │ Exercises          │
│     │ View Slides →         │ • Regularization   │
│     │                       │ • A1 (Due Week 4)  │
└─────────────────────────────────────────────────┘
```

### 新增内容

**作业 (Assignments):**
- ✅ A0: Python Warmup (Lecture 1)
  - 描述: Environment setup and basic NumPy
  - 截止: Week 1
  - 权重: 0-2%

- ✅ A1: Hedonic Pricing (Lecture 2)
  - 描述: OLS vs Ridge/Lasso regression
  - 截止: Week 4
  - 权重: 10%

**课堂练习 (In-Class Exercises):**
- ✅ Lecture 2: Regularization Problem
- ✅ Lecture 4: Cross-Validation Problem

---

## 📋 每讲内容分布

| Lecture | Slides | In-Class Exercise | Assignment |
|---------|--------|------------------|------------|
| 1 | ✓ Introduction & Supervised Learning | - | - |
| 2 | ✓ Regularization | ✓ Regularization Problem | ✓ A1 发布 |
| 3 | ✓ Trees & Forests | - | - |
| 4 | ✓ Cross-Validation | ✓ CV Problem | A1 截止 |
| 5 | ✓ DML | - | - |
| 6 | ✓ HTE | - | - |
| 7 | ✓ DAGs & SCM | - | - |
| 8 | ✓ IV & DML-IV | - | - |
| 9 | ✓ DiD & RDD | - | - |
| 10 | ✓ Policy Learning | - | - |

---

## 🎯 设计亮点

### 1. 视觉层次清晰
- **左侧 (Slides):** 大的讲次编号 + 标题 + 查看按钮
- **右侧 (Exercises):** 练习和作业卡片，带标签（In-Class / Assignment）

### 2. 信息丰富
每个作业显示：
- 标题和类型（标签）
- 简短描述
- 截止日期
- 权重

### 3. 响应式设计
- **桌面端 (>1024px):** 左右两栏
- **平板端 (768-1024px):** 左右两栏但紧凑
- **手机端 (<768px):** 单栏堆叠

### 4. 交互反馈
- 悬停卡片：轻微右移 + 背景变色
- 悬停按钮：上移 + 阴影增强
- 动画流畅自然

---

## 🎨 颜色系统

### 主色调
- 紫色渐变: `#667eea` → `#764ba2`
- 用于：标题、按钮、讲次编号

### 标签颜色
- **In-Class Exercise:** 蓝色 (`#e0f2fe` / `#0369a1`)
- **Assignment:** 黄色 (`#fef3c7` / `#92400e`)

### 背景
- 页面背景：紫色渐变
- 卡片背景：白色半透明 (95%)
- 资源卡片：浅灰色 (`#f8fafc`)

---

## 📁 新增文件

### presentations/ 目录
```
presentations/
├── exercises/                          # 新增
│   ├── Lecture02-Regularization-Problem.md
│   └── Lecture04-Cross-Validation-Problem.md
├── assignments/                        # 已存在，但现在被链接
│   ├── A0/
│   │   ├── README.md
│   │   ├── basic_stats.py
│   │   ├── test_basic_stats.py
│   │   └── ...
│   └── A1/
│       ├── README.md
│       ├── hw1_code.py
│       ├── hw1_report.md
│       └── ...
└── scripts/
    └── build-index.js                  # 大幅重写
```

---

## 🔧 技术更新

### build-index.js 改进

**新增功能:**
1. 读取并配置每讲的资源
2. 支持两种资源类型：
   - `inClassExercise` (课堂练习)
   - `assignments` (作业)
3. 自动生成两栏布局 HTML
4. 排除 `README.md` 被算作讲次

**配置示例:**
```javascript
const lectureResources = {
  2: {
    inClassExercise: {
      title: 'Regularization Problem',
      file: 'Lecture02-Regularization-Problem.md',
      type: 'exercise'
    },
    assignments: ['A1']
  },
  // ...
};

const assignmentInfo = {
  'A1': {
    title: 'A1: Hedonic Pricing',
    description: 'OLS vs Ridge/Lasso regression',
    due: 'Week 4',
    weight: '10%'
  }
};
```

---

## 📱 响应式断点

### Desktop (1400px+)
- 容器宽度: 1400px
- 两栏布局: 1fr 1fr (50/50)
- 完整间距和内边距

### Tablet (768px - 1024px)
- 两栏布局保持
- 间距略微减小
- 字体大小保持

### Mobile (<768px)
- 单栏堆叠布局
- Slides 在上，Exercises 在下
- 统计数字纵向排列
- 减小内边距

---

## 🌐 网站更新流程

### 自动部署 (GitHub Actions)
```
Push to main → GitHub Actions 触发
  ↓
1. npm ci (安装依赖)
  ↓
2. npm run build (生成 slides HTML)
  ↓
3. npm run build:index (生成新的 index.html)
  ↓
4. Deploy to GitHub Pages
  ↓
网站更新完成 (2-3 分钟)
```

### 访问新网站
- **URL:** https://jasminehao.com/econ6083-slides/
- **预计更新时间:** 推送后 2-3 分钟
- **清除缓存:** Ctrl+F5 (强制刷新)

---

## 🎯 未来扩展

### 可以轻松添加
1. **更多练习:** 在 `exercises/` 文件夹添加 .md 文件
2. **更多作业:** 在 `assignments/` 添加文件夹
3. **配置关联:** 在 `build-index.js` 的 `lectureResources` 中添加

### 示例：添加 Lecture 5 的练习
```javascript
// 1. 将练习文件复制到 exercises/
cp 课件/Lecture05-DML-Problem.md presentations/exercises/

// 2. 在 build-index.js 中配置
5: {
  inClassExercise: {
    title: 'DML Problem',
    file: 'Lecture05-DML-Problem.md',
    type: 'exercise'
  },
  assignments: []
}

// 3. 重新构建
npm run build:index

// 4. 推送到 GitHub
git add -A
git commit -m "Add Lecture 5 exercise"
git push
```

---

## ✅ 完成的改进

- [x] 左右两栏布局（Slides | Exercises）
- [x] 添加 A0 和 A1 作业页面
- [x] 添加 Lecture 2 和 4 的课堂练习
- [x] 响应式设计（适配手机/平板/桌面）
- [x] 美化资源卡片（标签、描述、元信息）
- [x] 修复 README.md 被算作讲次
- [x] 推送到 GitHub 自动部署

---

## 📊 统计

### 旧版本
- 单栏列表
- 仅显示 slides
- 简单链接

### 新版本
- 两栏布局
- Slides (10) + Exercises (2) + Assignments (2)
- 丰富的卡片界面
- 响应式设计
- 元信息显示（截止日期、权重）

---

## 🎉 总结

**全新的课程材料网站已上线！**

学生可以在同一页面看到：
- ✅ 每讲的 Slides
- ✅ 对应的课堂练习
- ✅ 相关的作业（带截止日期和权重）
- ✅ 清晰的视觉层次和交互反馈

**访问:** https://jasminehao.com/econ6083-slides/

等待 2-3 分钟后自动部署完成！

---

**设计完成时间:** 2026-02-09 18:40
