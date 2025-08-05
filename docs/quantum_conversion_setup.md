# TensorCircuit-NG 量子态转换功能开发文档

## 概述

本文档记录了在 TensorCircuit-NG 项目中新增的量子态转换功能，包括 TeNPy、Quimb 和 TensorNetwork 包之间的互相转换函数。这些函数使得用户可以在不同的量子计算包之间无缝转换量子态和算符。

---

## 新增功能

### 1. 转换函数

#### 1.1 从外部包转换到 TensorCircuit

- **`tenpy2qop(tenpy_obj: Union[MPS, MPO]) -> QuOperator`**
  - 将 TeNPy 的 MPS 或 MPO 对象转换为 TensorCircuit 的 QuOperator
  - 支持有限边界条件（finite boundary conditions）
  - 正确处理轴序和边界条件以兼容 `eval_matrix` 方法

#### 1.2 从 TensorCircuit 转换到外部包

- **`qop2tenpy(qop: QuOperator) -> Union[MPS, MPO]`**
  - 将 TensorCircuit 的 QuOperator 转换为 TeNPy 的 MPS 或 MPO 对象
  - 自动检测是否为 MPS（无输入边）或 MPO（有输入边）
  - 处理边界填充和维度匹配

- **`qop2quimb(qop: QuOperator) -> Union[MatrixProductState, MatrixProductOperator]`**
  - 将 TensorCircuit 的 QuOperator 转换为 Quimb 的 MPS 或 MPO 对象
  - 自动生成正确的索引标签
  - 支持 TensorNetwork 到 Quimb 的接口转换

- **`qop2tn(qop: QuOperator) -> Union[FiniteMPS, FiniteMPO]`**
  - 将 TensorCircuit 的 QuOperator 转换为 TensorNetwork 的 MPS 或 MPO 对象
  - 正确处理张量轴的排列和边界条件

---

## 技术实现细节

### 数据结构对应关系

| 包名 | MPS 类型 | MPO 类型 | 特点 |
|------|----------|----------|------|
| TensorCircuit | QuOperator | QuOperator | 统一的量子算符表示 |
| TeNPy | MPS | MPO | 基于 `np_conserved` 张量 |
| Quimb | MatrixProductState | MatrixProductOperator | 基于索引标签的张量网络 |
| TensorNetwork | FiniteMPS | FiniteMPO | 原生 numpy/backend 张量 |

### 关键技术挑战

1. **轴序转换**：不同包对张量轴的排列有不同的约定
2. **边界处理**：处理 MPS/MPO 的左右边界条件
3. **节点排序**：确保转换后的张量顺序正确
4. **维度兼容性**：处理不同包的维度表示差异

---

## 依赖关系

### 必需依赖
```python
from tenpy.networks import MPO, MPS, Site
import tenpy.linalg.np_conserved as npc
from tenpy.linalg import LegCharge
import tensornetwork as tn
import quimb.tensor as qtn
```

### 环境配置
在 `requirements/requirements-hyx.txt` 中添加了以下依赖：
```
physics-tenpy
quimb
```

---

## 测试覆盖

### 测试函数列表

#### 基础转换测试

1. **`test_tenpy2qop(backend)`**
   - 测试 TeNPy 到 TensorCircuit 的转换
   - 验证 TFI 链哈密顿量的数值正确性
   - 验证产品态的转换准确性

2. **`test_qop2tenpy(backend)`**
   - 测试 TensorCircuit 到 TeNPy 的转换
   - 验证 MPS 和 MPO 的转换
   - 检验边界条件处理

3. **`test_qop2quimb(backend)`**
   - 测试 TensorCircuit 到 Quimb 的转换
   - 验证张量网络结构的正确性
   - 检验索引标签的一致性

4. **`test_qop2tn(backend)`**
   - 测试 TensorCircuit 到 TensorNetwork 的转换
   - 验证轴排列的正确性
   - 检验数值精度

#### 往返测试（Roundtrip Tests）

5. **`test_tenpy_roundtrip(backend)`**
   - 测试 TeNPy → QuOperator → TeNPy 的完整往返
   - 验证 MPO 和 MPS 的数值一致性
   - 检验边界条件和维度处理

6. **`test_quimb_roundtrip(backend)`**
   - 测试 Quimb → QuOperator → Quimb 的完整往返
   - 验证张量网络的结构保持
   - 检验索引标签的正确性

（没有包含tensornetwork的往返测试是因为已有的tn2qop函数并不支持MPS类的转换，而qop2tn函数支持MPS类的转换）

### 测试策略

- **数值验证**：使用已知的物理模型（如 TFI 链）验证转换正确性
- **往返测试**：确保 A→B→A 的转换保持数值一致性
- **边界情况**：测试单量子比特、边界条件等特殊情况
- **多后端支持**：在 numpy、tensorflow、jax 后端下测试

---

## 使用示例

### 基本用法

```python
import tensorcircuit as tc
from tenpy.models.tf_ising import TFIChain

# 创建 TeNPy 模型
model_params = {"L": 4, "J": 1.0, "g": -1.0, "bc_MPS": "finite"}
model = TFIChain(model_params)

# TeNPy MPO 转换为 TensorCircuit QuOperator
tenpy_mpo = model.H_MPO
tc_qop = tc.quantum.tenpy2qop(tenpy_mpo)

# TensorCircuit QuOperator 转换为其他格式
quimb_mpo = tc.quantum.qop2quimb(tc_qop)
tn_mpo = tc.quantum.qop2tn(tc_qop)
back_to_tenpy = tc.quantum.qop2tenpy(tc_qop)
```

### 高级用法

```python
# 验证转换正确性
original_matrix = tenpy_mpo.to_matrix()
converted_matrix = tc_qop.eval_matrix()
np.testing.assert_allclose(original_matrix, converted_matrix, atol=1e-10)

# 在不同格式间进行计算
result_quimb = quimb_mpo.to_dense()
result_tn = tn_mpo.to_dense()
```

---

## 开发历程

### 初始需求
- 项目需要在不同量子计算包之间进行互操作
- 特别需要支持 TeNPy 的 MPS/MPO 格式转换
- 用户希望能够利用不同包的优势（如 TeNPy 的 DMRG、Quimb 的可视化等）

### 实现过程

1. **调研阶段**：分析各包的数据结构和 API
2. **原型开发**：实现基础的转换功能
3. **测试完善**：添加全面的测试覆盖
4. **文档编写**：提供使用说明和API文档

### 遇到的问题及解决方案

1. **TeNPy 数据结构理解**：
   - 问题：初期对 `tenpy_obj.A` 和 `tenpy_obj._B` 的区别理解有误
   - 解决：通过查阅源码发现应使用 `_B` 获取张量列表

2. **轴序问题**：
   - 问题：不同包对 MPO 张量轴的排列约定不同
   - 解决：实现了动态轴转置以适配各包的要求

3. **边界条件处理**：
   - 问题：有限 MPS/MPO 的边界处理复杂
   - 解决：通过分析端点节点和物理边来正确排序

---

## 性能考虑

### 内存效率
- 转换过程尽可能避免不必要的张量复制
- 使用视图（view）操作减少内存占用

### 计算效率
- 转换函数的时间复杂度为 O(n)，其中 n 为张量数量
- 大部分操作为内存重排，计算开销较小

### 数值精度
- 所有转换保持原始精度，默认使用 double precision
- 测试中验证数值误差在 1e-10 范围内

---

## 未来改进计划

### 短期目标
- [ ] 支持更多的边界条件类型（如周期边界条件）
- [ ] 添加对称性保持的转换
- [ ] 优化大规模张量网络的转换性能

### 长期目标
- [ ] 支持更多量子计算包（如 ITensor、ALPS 等）
- [ ] 实现自动格式检测和智能转换
- [ ] 添加可视化工具辅助调试

### 代码质量改进
- [ ] 增加类型注解的完整性
- [ ] 改进错误处理和用户提示
- [ ] 添加性能基准测试

---

## 贡献指南

### 代码规范
- 遵循项目的代码风格（使用 black 格式化）
- 添加完整的文档字符串
- 确保类型注解的准确性

### 测试要求
- 新功能必须包含相应的测试
- 测试覆盖率应达到 90% 以上
- 包含边界情况和错误处理测试

### 文档要求
- 更新 API 文档
- 提供使用示例
- 记录已知限制和注意事项

---

## 相关资源

### 官方文档
- [TeNPy Documentation](https://tenpy.readthedocs.io/)
- [Quimb Documentation](https://quimb.readthedocs.io/)
- [TensorNetwork Documentation](https://tensornetwork.readthedocs.io/)

### 相关论文
- Matrix Product States and their applications
- Tensor Network Methods for Quantum Many-Body Systems
- Efficient simulation of quantum systems using tensor networks

### 代码仓库
- [TensorCircuit-NG](https://github.com/Charlespkuer/tensorcircuit-ng)
- [原始 TensorCircuit](https://github.com/refraction-ray/tensorcircuit)
