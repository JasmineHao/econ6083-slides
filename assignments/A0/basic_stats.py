"""
A0: 热身作业 - 基础统计函数

这是一个非常简单的热身作业，帮助你熟悉：
1. GitHub Classroom 工作流
2. Python 基础
3. 单元测试

不要紧张！这个作业不计分（或只占极少分数），目的是让你熟悉流程。
"""

import numpy as np


def calculate_mean(numbers):
    """
    计算一组数字的平均值

    参数:
        numbers (list or np.ndarray): 一组数字

    返回:
        float: 平均值

    示例:
        >>> calculate_mean([1, 2, 3, 4, 5])
        3.0
        >>> calculate_mean([10, 20, 30])
        20.0
    """
    # TODO: 在这里实现你的代码
    # 提示: 可以使用 sum() 和 len()，或者 np.mean()

    pass  # 删除这一行，写入你的代码


def calculate_std(numbers):
    """
    计算一组数字的标准差

    参数:
        numbers (list or np.ndarray): 一组数字

    返回:
        float: 标准差

    示例:
        >>> calculate_std([1, 2, 3, 4, 5])
        1.4142135623730951
    """
    # TODO: 在这里实现你的代码
    # 提示: 可以使用 np.std()

    pass  # 删除这一行，写入你的代码


def normalize_array(numbers):
    """
    将一组数字标准化 (z-score normalization)

    公式: (x - mean) / std

    参数:
        numbers (list or np.ndarray): 一组数字

    返回:
        np.ndarray: 标准化后的数组

    示例:
        >>> normalize_array([1, 2, 3, 4, 5])
        array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    """
    # TODO: 在这里实现你的代码
    # 提示: 使用前面写的 calculate_mean 和 calculate_std
    # 或者直接用 numpy 的函数

    pass  # 删除这一行，写入你的代码


# ===== 额外挑战题（可选，不影响分数）=====

def calculate_correlation(x, y):
    """
    计算两组数字的相关系数

    这是一个可选的挑战题，如果你想多练习可以完成。
    不完成也不影响分数！

    参数:
        x (list or np.ndarray): 第一组数字
        y (list or np.ndarray): 第二组数字

    返回:
        float: 相关系数 (范围: -1 到 1)

    提示: 可以使用 np.corrcoef()
    """
    # 可选题 - 如果你想挑战的话
    pass


if __name__ == "__main__":
    # 你可以在这里测试你的函数
    print("测试 calculate_mean:")
    test_numbers = [1, 2, 3, 4, 5]
    print(f"平均值: {calculate_mean(test_numbers)}")

    print("\n测试 calculate_std:")
    print(f"标准差: {calculate_std(test_numbers)}")

    print("\n测试 normalize_array:")
    print(f"标准化: {normalize_array(test_numbers)}")

    # 运行这个文件，看看你的函数是否工作正常
    # 然后运行 pytest test_basic_stats.py 进行正式测试
