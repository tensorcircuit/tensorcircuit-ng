# 天衍（TianYan）量子云平台接入说明

## 概述

本文档介绍 TensorCircuit-NG 对[中电信天衍量子计算云平台](https://qc.zdxlz.com)的接入支持。

## 平台特性

- **提供商**: `tianyan`
- **SDK**: `cqlib`（天衍官方 Python SDK）
- **指令集**: QCIS（原生支持）
- **设备类型**:
  - 超导量子计算机（真实量子硬件）
  - 量子仿真器（软件模拟）

## 快速开始

### 1. 安装依赖

```bash
pip install "tensorcircuit-ng[cloud]"
pip install "cqlib>=1.3.10,<1.4"
```

天衍官方 SDK `cqlib` 要求 Python 3.10 或更高版本，并会安装 NumPy
2.1.2 或更高版本。当前实现验证的 SDK 范围为 `cqlib>=1.3.10,<1.4`。
`cqlib` 不属于 TensorCircuit-NG 的通用 `cloud` extra，需要按上面的命令
单独安装；TensorCircuit-NG 本身仍支持 Python 3.9。

### 2. 获取登录密钥

访问[天衍量子计算云平台](https://qc.zdxlz.com)注册账号并获取 SDK 登录密钥。不要把密钥写入脚本、Notebook、日志或版本控制。

Windows PowerShell 下可以把密钥保存到当前 Conda 环境：

```powershell
conda activate <env>
conda env config vars set TIANYAN_LOGIN_KEY="your_login_key"
conda deactivate
conda activate <env>
```

`set`、`TIANYAN_LOGIN_KEY` 和 `=` 之间不要添加空格。重新激活环境后变量才会生效。

### 3. 基本使用

```python
import os

import tensorcircuit as tc

login_key = os.getenv("TIANYAN_LOGIN_KEY")
if not login_key:
    raise RuntimeError("请先设置 TIANYAN_LOGIN_KEY 环境变量")

tc.cloud.apis.set_provider("tianyan")
tc.cloud.apis.set_token(login_key, provider="tianyan", cached=False)

# 列出设备
devices = tc.cloud.apis.list_devices()
print(devices)

# 选择设备（仿真器或真实量子机）
device = tc.cloud.apis.get_device("tianyan::tianyan_sw")

# 创建电路
c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)
c.measure_instruction(0, 1)

# 提交任务
task = tc.cloud.apis.submit_task(circuit=c, device=device, shots=1000)

# 获取结果
counts = task.results(blocked=True)
print(counts)
```

## API 参考

### 设备管理

| API | 说明 |
|-----|------|
| `tc.cloud.apis.set_provider("tianyan")` | 设置当前 provider |
| `tc.cloud.apis.set_token(key, provider="tianyan")` | 设置登录密钥 |
| `tc.cloud.apis.list_devices()` | 列出所有可用设备 |
| `tc.cloud.apis.get_device("tianyan::DEVICE_NAME")` | 获取特定设备 |
| `device.list_properties()` | 获取设备属性（拓扑、校准数据） |
| `device.topology()` | 获取拓扑边列表 |
| `device.native_gates()` | 获取支持的 native gates |

### 任务提交

| API | 说明 |
|-----|------|
| `tc.cloud.apis.submit_task(circuit=c, device=d, shots=1000)` | 提交单个任务 |
| `tc.cloud.apis.submit_task(circuit=[c1, c2], ...)` | 批量提交 |
| `tc.cloud.apis.submit_task(source="QCIS string", ...)` | 直接提交 QCIS |
| `tc.cloud.apis.submit_task(source=qasm, lang="OPENQASM", ...)` | 转换并提交 OpenQASM 2 源码 |

### 任务管理

| API | 说明 |
|-----|------|
| `task.results(blocked=True)` | 获取结果（阻塞等待） |
| `task.details()` | 获取任务详情 |
| `task.state()` | 获取任务状态 |
| `task.resubmit()` | 重新提交任务 |

天衍 SDK 当前没有任务列表和任务取消接口，因此
`tc.cloud.apis.list_tasks(provider="tianyan")` 和
`tc.cloud.apis.remove_task(...)` 会明确抛出 `NotImplementedError`。

## 特色功能

### 拓扑校验

Provider 不会自动重映射比特。提交到真实量子机的电路必须已经满足设备拓扑约束：
请先使用 `tensorcircuit.compiler` 等工具完成编译和映射。对于 TensorCircuit 电路，
provider 会在提交前校验拓扑兼容性，若某个门作用在不可用比特或不连通的物理比特对上，
会抛出 `ValueError`，且不会提交任何任务。对于 qiskit 电路和直接提供的
QCIS/OpenQASM 源码，拓扑兼容性由用户自行保证。

一个简单做法是直接在 `device.topology()` 返回的连通物理比特对上构建电路：

```python
run_hardware = os.getenv("TIANYAN_RUN_HARDWARE") == "1"
if run_hardware:
    device = tc.cloud.apis.get_device("tianyan::tianyan176")
    q1, q2 = sorted(device.topology()[0])
    c = tc.Circuit(q2 + 1)
    c.h(q1)
    c.cx(q1, q2)
    c.measure_instruction(q1, q2)
    task = tc.cloud.apis.submit_task(circuit=c, device=device, shots=100)
    print(task.details()["source"])
else:
    print("已跳过真实设备提交")
```

### 任务详情增强

任务详情包含：

- `source`: 提交的 QCIS 源码
- `state`: 任务状态
- `shots`: 测量次数
- `results`: 测量结果 counts

### 批量提交

```python
circuits = [c1, c2, c3]
tasks = tc.cloud.apis.submit_task(circuit=circuits, device=device, shots=100)
for t in tasks:
    print(t.results(blocked=True))
```

### 直接 QCIS 提交

```python
qcis = "H Q0\nH Q1\nCZ Q0 Q1\nH Q1\nM Q0\nM Q1"
task = tc.cloud.apis.submit_task(source=qcis, device=device, shots=100)
```

## 设备列表

| 设备名 | 类型 | 说明 |
|--------|------|------|
| `tianyan_sw` | 仿真器 | 全振幅仿真 |
| `tianyan_s` | 仿真器 | 单振幅仿真 |
| `tianyan_tn` | 仿真器 | 张量网络仿真 |
| `tianyan176` | 真实量子机 | 176 比特超导量子计算机 |
| `tianyan24` | 真实量子机 | 24 比特超导量子计算机 |

设备名称和可用状态以
`tc.cloud.apis.list_devices(provider="tianyan")` 的实时返回为准。

## 注意事项

1. **仿真器 vs 真实量子机**: 仿真器无拓扑限制，结果接近理想；真实量子机有拓扑限制和噪声
2. **拓扑校验**: 仅对真实量子机启用，仿真器跳过；不兼容的 TensorCircuit 电路会在提交前抛出 `ValueError`
3. **单比特线路**: 部分真实量子机可能不支持单比特线路
4. **Shots**: 真实量子机不宜设置过大，避免额度消耗
5. **硬件开关**: 示例和 Notebook 默认不提交真实设备；显式设置 `TIANYAN_RUN_HARDWARE=1` 后才会运行硬件部分
6. **测量语义**: 测量一律按记录顺序作为末端测量提交，不保留中途测量（mid-circuit measurement）语义；依赖中途测量结果的线路请勿直接提交

## 相关文件

- `tensorcircuit/cloud/tianyan.py`: 天衍 provider 实现
- `tensorcircuit/cloud/apis.py`: 统一 API 入口
- `examples/tianyan_cloud_demo.py`: 完整示例脚本
- `docs/source/tutorials/tianyan_cloud_cn.ipynb`: 中文 Jupyter 教程
- `docs/source/tutorials/tianyan_cloud_reference_cn.md`: 本 API 参考文档

## 故障排查

### `cqlib` 未安装

```bash
pip install "cqlib>=1.3.10,<1.4"
```

### 登录密钥无效

检查环境变量是否生效以及密钥是否正确，或访问[天衍量子计算云平台](https://qc.zdxlz.com)重新获取。

### 拓扑校验失败

- 先使用 `tensorcircuit.compiler` 等工具将电路编译并映射到设备拓扑
- 查看 `device.list_properties()["links"]` 了解可用耦合关系
- 直接在 `device.topology()` 返回的连通物理比特上构建电路

### 任务运行失败

- 检查所选比特是否支持当前线路（查看拓扑结构）
- 检查设备状态是否为 `running`

## 更多信息

- [天衍平台文档](https://cqlib.readthedocs.io/)
- [TensorCircuit 文档](https://tensorcircuit.readthedocs.io/)
