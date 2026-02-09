"""
A0 测试文件

这个文件包含自动测试，用于检查你的代码是否正确。

运行方法:
    pytest test_basic_stats.py -v

不要修改这个文件！
"""

import pytest
import numpy as np
from basic_stats import calculate_mean, calculate_std, normalize_array


class TestCalculateMean:
    """测试 calculate_mean 函数"""

    def test_simple_list(self):
        """测试简单的列表"""
        result = calculate_mean([1, 2, 3, 4, 5])
        assert abs(result - 3.0) < 0.0001, "平均值应该是 3.0"

    def test_all_same(self):
        """测试所有数字相同"""
        result = calculate_mean([5, 5, 5, 5])
        assert abs(result - 5.0) < 0.0001, "相同数字的平均值应该是那个数字本身"

    def test_negative_numbers(self):
        """测试负数"""
        result = calculate_mean([-1, -2, -3, -4])
        assert abs(result - (-2.5)) < 0.0001, "负数也应该能正确计算"

    def test_numpy_array(self):
        """测试 numpy 数组"""
        arr = np.array([10, 20, 30, 40])
        result = calculate_mean(arr)
        assert abs(result - 25.0) < 0.0001, "应该能处理 numpy 数组"


class TestCalculateStd:
    """测试 calculate_std 函数"""

    def test_simple_list(self):
        """测试简单的列表"""
        result = calculate_std([1, 2, 3, 4, 5])
        expected = np.std([1, 2, 3, 4, 5])
        assert abs(result - expected) < 0.0001, f"标准差应该是 {expected}"

    def test_all_same(self):
        """测试所有数字相同（标准差应该为0）"""
        result = calculate_std([5, 5, 5, 5])
        assert abs(result - 0.0) < 0.0001, "相同数字的标准差应该是 0"

    def test_larger_variance(self):
        """测试较大的方差"""
        result = calculate_std([1, 100, 1, 100])
        expected = np.std([1, 100, 1, 100])
        assert abs(result - expected) < 0.0001, "应该能处理大方差的数据"


class TestNormalizeArray:
    """测试 normalize_array 函数"""

    def test_simple_normalization(self):
        """测试简单的标准化"""
        result = normalize_array([1, 2, 3, 4, 5])

        # 标准化后均值应该接近0
        mean_after = np.mean(result)
        assert abs(mean_after) < 0.0001, "标准化后均值应该接近 0"

        # 标准化后标准差应该接近1
        std_after = np.std(result)
        assert abs(std_after - 1.0) < 0.0001, "标准化后标准差应该接近 1"

    def test_returns_array(self):
        """测试返回类型"""
        result = normalize_array([1, 2, 3])
        assert isinstance(result, np.ndarray), "应该返回 numpy 数组"

    def test_length_preserved(self):
        """测试长度保持不变"""
        original = [10, 20, 30, 40, 50]
        result = normalize_array(original)
        assert len(result) == len(original), "标准化后长度应该不变"


# ===== 额外的友好测试（帮助调试）=====

def test_functions_exist():
    """检查所有必需的函数是否存在"""
    from basic_stats import calculate_mean, calculate_std, normalize_array
    # 如果能导入，这个测试就通过
    assert True


def test_basic_functionality():
    """一个综合测试，检查基本功能"""
    try:
        # 简单的烟雾测试
        mean_result = calculate_mean([1, 2, 3])
        assert mean_result is not None, "calculate_mean 不应该返回 None"

        std_result = calculate_std([1, 2, 3])
        assert std_result is not None, "calculate_std 不应该返回 None"

        norm_result = normalize_array([1, 2, 3])
        assert norm_result is not None, "normalize_array 不应该返回 None"

        print("✓ 基本功能测试通过！")
    except Exception as e:
        pytest.fail(f"基本功能测试失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
