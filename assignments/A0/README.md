# A0: 热身作业 - Python基础 + 课程反馈

> **作业性质**：不计分（或仅占1-2分参与分）
>
> **目的**：
> 1. 确认 Python 环境正常
> 2. 了解你的背景和期望,以便调整课程
> 3. 熟悉作业提交流程

---

## 🎯 作业内容

### Part 1: 写三个简单函数（30分钟）

在 `basic_stats.py` 中完成三个基础统计函数：

1. **`calculate_mean(numbers)`** - 计算平均值
2. **`calculate_std(numbers)`** - 计算标准差
3. **`normalize_array(numbers)`** - 标准化数组

这些函数**非常简单**，主要是确认你能：
- 使用 NumPy 基础函数
- 理解函数的输入输出
- 运行单元测试

### Part 2: 填写反馈问卷（10分钟）

在 `feedback.md` 中填写：
- 你的编程背景
- 对课程的期望
- 需要的支持

**重要**：问卷不影响成绩，请如实填写！

---

## 🚀 开始步骤

### 1. 下载作业文件

从 Moodle 课程页面下载作业压缩包（A0.zip），解压到你的工作目录。

### 2. 安装依赖

打开终端（Windows 使用 PowerShell 或 CMD），进入作业文件夹：

```bash
cd path/to/A0

# 安装 NumPy 和 pytest
pip install numpy pytest

# 或者使用 requirements.txt
pip install -r requirements.txt
```

### 3. 完成代码

打开 `basic_stats.py`，找到标记为 `# TODO` 的地方，填写你的代码。

**提示**：
```python
# 计算平均值
def calculate_mean(numbers):
    return np.mean(numbers)  # 或者 sum(numbers) / len(numbers)

# 计算标准差
def calculate_std(numbers):
    return np.std(numbers)

# 标准化
def normalize_array(numbers):
    numbers = np.array(numbers)
    mean = np.mean(numbers)
    std = np.std(numbers)
    return (numbers - mean) / std
```

### 4. 测试你的代码

```bash
# 运行你自己的测试
python basic_stats.py

# 运行自动测试
pytest test_basic_stats.py -v
```

如果所有测试都通过，说明你的代码是正确的！

### 5. 填写反馈问卷

打开 `feedback.md`，回答问题。

### 6. 提交作业

将以下文件打包成 zip 文件：
- `basic_stats.py`（你完成的代码）
- `feedback.md`（填写完的问卷）

**命名格式**：`A0_你的学号_你的姓名.zip`

例如：`A0_2024001_张三.zip`

然后在 Moodle 上传这个 zip 文件。

---

## ✅ 评分标准（如果计分的话）

| 部分 | 分数 | 说明 |
|------|------|------|
| 代码正确性 | 50% | 通过自动测试 |
| 反馈问卷 | 50% | 完整填写即可 |

**但其实**：这个作业主要是熟悉流程，只要提交了就有分！

---

## 🆘 遇到问题？

### 常见问题

**Q: 我不会安装 Python 怎么办？**

A:
1. Python: 访问 [python.org](https://python.org)，下载 3.8+ 版本
2. 安装时勾选 "Add Python to PATH"
3. 或者来 office hours，我们一起解决！

**Q: 测试失败了怎么办？**

A:
1. 看错误信息，通常会告诉你哪里错了
2. 用 `print()` 输出中间结果，调试你的代码
3. 检查是否返回了正确的类型（比如应该返回 float，不是 None）
4. 来 office hours 寻求帮助

**Q: 我完全不会 NumPy 怎么办？**

A:
- 这三个函数只需要用 `np.mean()`, `np.std()` 和基础算术
- 查看 [NumPy 快速入门](https://numpy.org/doc/stable/user/quickstart.html)
- 或者直接用纯 Python：`sum(numbers) / len(numbers)`

**Q: 如何在 Windows 上打包 zip 文件？**

A:
1. 选中 `basic_stats.py` 和 `feedback.md` 两个文件
2. 右键 → "发送到" → "压缩(zipped)文件夹"
3. 重命名为 `A0_学号_姓名.zip`

**Q: 反馈问卷写错了可以改吗？**

A: 当然！重新上传即可，Moodle 会保留最新版本。

---

## 💡 这个作业真的很简单！

如果你觉得这个作业太简单了，说明你准备得很好！后面的作业会逐渐增加难度。

如果你觉得有点难，不要担心：
- 这个作业不影响总成绩（或只占极少分数）
- 目的是帮你熟悉流程
- 我们会根据大家的反馈调整后续课程
- 有 office hours 和助教支持

---

## 📚 学习资源

如果你想提前预习：

**Python 基础**：
- [Python 官方教程](https://docs.python.org/3/tutorial/)
- [菜鸟教程 - Python3](https://www.runoob.com/python3/python3-tutorial.html)

**NumPy 入门**：
- [NumPy 官方教程](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 中文文档](https://www.numpy.org.cn/)

---

## 🎉 完成后

提交作业后，我们会运行自动测试并给出反馈。通常在截止日期后一周内会有结果。

不要担心测试失败！如果有问题，我们会在反馈中告诉你如何改进。

**期待看到你的提交！** 如果有任何问题，随时联系我。

---

**截止日期**: [老师填写]
**Office Hours**: [老师填写具体时间]
**Email**: [老师填写邮箱]
**Moodle 课程页面**: [链接]
