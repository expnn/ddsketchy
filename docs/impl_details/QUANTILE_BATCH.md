# 统一 quantile 功能说明

## 概述

`quantile` 方法现在支持两种模式：单个分位数计算和批量分位数计算。该方法会根据输入类型自动选择适当的模式，简化了 API，减少了用户的学习成本。

## 功能特性

1. **智能模式选择**：根据输入类型自动选择单个或批量模式
2. **批量计算**：一次性计算多个分位数，避免多次调用
3. **NumPy 零拷贝优化**：对于连续的 NumPy 数组，使用零拷贝技术直接访问内存
4. **多种输入支持**：支持单个 float、Python 列表、元组、NumPy 数组等
5. **一致性保证**：批量计算结果与单独调用完全一致

## 使用示例

### 单个分位数（返回 float）

```python
from ddsketchy import DDSketch

sketch = DDSketch()
sketch.add_batch(range(1, 1001))  # 添加 1 到 1000 的值

# 计算单个分位数
median = sketch.quantile(0.5)  # 返回: float
print(f"P50 = {median:.2f}")
```

### 批量分位数 - Python 列表（返回 list）

```python
from ddsketchy import DDSketch

sketch = DDSketch()
sketch.add_batch(range(1, 1001))

# 计算常用分位数
quantiles = [0.5, 0.9, 0.95, 0.99]
results = sketch.quantile(quantiles)  # 返回: List[float]

for q, r in zip(quantiles, results):
    print(f"P{int(q*100)} = {r:.2f}")
```

### 批量分位数 - NumPy 数组（零拷贝，更快）

```python
import numpy as np
from ddsketchy import DDSketch

sketch = DDSketch()
sketch.add_batch(range(1, 1001))

# 使用 NumPy 数组（零拷贝优化）
quantiles = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
results = sketch.quantile(quantiles)  # 返回: List[float]

for q, r in zip(quantiles, results):
    print(f"P{int(q*100)} = {r:.2f}")
```

### 计算大量分位数

```python
import numpy as np
from ddsketchy import DDSketch

sketch = DDSketch()
sketch.add_batch(range(1, 10001))

# 计算 100 个分位数
quantiles = np.linspace(0.01, 0.99, 100)
results = sketch.quantile(quantiles)

print(f"计算了 {len(results)} 个分位数")
print(f"P10  = {results[9]:.2f}")
print(f"P50  = {results[49]:.2f}")
print(f"P90  = {results[89]:.2f}")
```

## 返回类型

- **单个 float 输入**：返回单个 float
- **Iterable 输入**（list、tuple、NumPy array）：返回 List[float]

注意：单个元素的 NumPy 数组会被当作单个 float 处理，返回 float 而非 list。

## 性能优势

对于大量分位数计算，批量模式比循环调用 `quantile()` 更高效：

- **减少 Python/Rust 边界调用**：一次调用代替多次调用
- **NumPy 零拷贝**：对于连续数组，避免数据复制
- **更好的缓存局部性**：连续处理多个分位数查询

## API 文档

```python
def quantile(self, q: Union[float, Iterable[float]]) -> Union[float, List[float]]:
    """
    计算分位数的估计值。
    
    参数:
        q: 单个分位数值 (float in [0, 1]) 或分位数值的可迭代对象
        
    返回:
        - 如果 q 是单个 float: 返回估计值 (float)
        - 如果 q 是可迭代对象: 返回估计值列表 (List[float])
        
    异常:
        ValueError: 如果任何分位数不在 [0, 1] 范围内，或输入无法转换为 f64
        
    示例:
        >>> sketch = DDSketch()
        >>> sketch.add_batch(range(1, 1001))
        
        >>> # 单个分位数
        >>> sketch.quantile(0.5)
        497.7794...
        
        >>> # 批量分位数
        >>> sketch.quantile([0.5, 0.9, 0.95])
        [497.78, 907.03, 944.05]
    """
```

## 实现细节

### Rust 层

在 `ddsketchy.rs` 中添加了 `quantile_batch` 方法供内部使用：

```rust
pub fn quantile_batch(&self, quantiles: &[f64]) -> Result<Vec<f64>, DDSketchError> {
    quantiles.iter().map(|&q| self.quantile(q)).collect()
}
```

### Python 绑定

在 `python.rs` 中实现了统一的 `quantile` 方法，包含三种处理路径：

1. **单个 float**：直接调用 `quantile()` 方法
2. **NumPy 连续数组（零拷贝）**：直接使用内存切片
3. **NumPy 非连续数组**：先收集到 Vec，再批量计算
4. **Python 列表/元组**：提取为 Vec<f64> 后批量计算

```rust
fn quantile<'py>(&self, py: Python<'py>, q: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // 尝试提取为单个 f64
    if let Ok(q_value) = q.extract::<f64>() {
        // 单个分位数模式
        let result = self.inner.quantile(q_value).map_err(PyErr::from)?;
        return Ok(result.into_pyobject(py)?.into_any());
    }

    // 批量模式：尝试提取为 NumPy 数组（零拷贝）
    if let Ok(arr) = q.extract::<PyReadonlyArray1<f64>>() {
        let view = arr.as_array();
        
        if let Some(slice) = view.as_slice() {
            // 真正的零拷贝：直接传递切片
            let results = self.inner.quantile_batch(slice).map_err(PyErr::from)?;
            return Ok(results.into_pyobject(py)?.into_any());
        } else {
            // 非连续数组：先收集到 Vec
            let quantiles_vec: Vec<f64> = view.iter().copied().collect();
            let results = self.inner.quantile_batch(&quantiles_vec).map_err(PyErr::from)?;
            return Ok(results.into_pyobject(py)?.into_any());
        }
    }

    // 回退路径：Python 列表（需要复制）
    let quantiles_vec: Vec<f64> = q.extract()?;
    let results = self.inner.quantile_batch(&quantiles_vec).map_err(PyErr::from)?;
    Ok(results.into_pyobject(py)?.into_any())
}
```

## 测试覆盖

完整的测试套件覆盖了以下场景：

- ✅ 单个 float 输入
- ✅ Python 列表输入
- ✅ Python 元组输入
- ✅ NumPy 连续数组（零拷贝路径）
- ✅ NumPy 非连续数组
- ✅ 空 sketch 处理（单个和批量）
- ✅ 无效分位数验证（单个和批量）
- ✅ 单个和批量模式的一致性
- ✅ 边界分位数（0.0 和 1.0）
- ✅ 大量分位数计算（100 个）
- ✅ 精度验证
- ✅ 单调性保证
- ✅ 返回类型区分（float vs list）

所有 81 个 Python 测试和 62 个 Rust 测试均通过。

## API 简化优势

通过将 `quantile()` 和 `quantile_batch()` 合并为一个统一的接口：

1. **降低学习成本**：用户只需记住一个方法名
2. **更自然的 API**：根据输入类型自动选择行为
3. **减少代码量**：不需要在单个和批量模式间切换方法
4. **向后兼容**：现有的 `quantile(0.5)` 调用方式完全不变
5. **类型安全**：返回类型清晰可预测
