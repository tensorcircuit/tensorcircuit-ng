# TensorCircuit-NG 开发笔记

## 重要修正说明

### tenpy_transform.md 相关修正
- **问题**：遍历 `tenpy_obj.A` 获取张量列表
- **修正**：实际应为 `tenpy_obj._B`

---

## 环境配置

### 新增依赖库配置
在文件 `tensorcircuit-ng/tensorcircuit/quantum.py` 中添加了以下导入：

### 2025年7月28日

#### 最终解决方案

1. **依赖关系分析**：
   - 问题根源：`cirq → cirq-rigetti → pyquil → qcs-sdk-python` 依赖链导致安装问题
   - 解决方案：询问老师后确认去掉 `cirq-rigetti` 不会影响项目

2. **成功安装的包**：
   - ✅ `pip install -r requirements/requirements-dev.txt`.png)

```python
from tenpy.networks import MPO, MPS, Site
import tenpy.linalg.np_conserved as npc
import tensornetwork as tn
from quimb.tensor import MatrixProductOperator
```

**说明**：已在新建文件 `tensorcircuit-ng/requirements/requirements-hyx.txt` 中添加了相应的库依赖。

---

## 数据结构分析

### TeNPy 中的 MPO（Matrix Product Operator）数据结构

MPO 是 MPS 向算符的推广。其图形表示如下：

```
Matrix product operator (MPO).

An MPO is the generalization of an :class:`~tenpy.networks.mps.MPS` to operators. Graphically::

    |      ^        ^        ^
    |      |        |        |
    |  ->- W[0] ->- W[1] ->- W[2] ->- ...
    |      |        |        |
    |      ^        ^        ^

So each 'matrix' has two physical legs ``p, p*`` instead of just one,
i.e. the entries of the 'matrices' are local operators.
Valid boundary conditions of an MPO are the same as for an MPS
(i.e. ``'finite' | 'segment' | 'infinite'``).
(In general, you can view the MPO as an MPS with larger physical space and bring it into
canonical form. However, unlike for an MPS, this doesn't simplify calculations.
Thus, an MPO has no `form`.)

We use the following label convention for the `W` (where arrows indicate `qconj`)::

    |            p*
    |            ^
    |            |
    |     wL ->- W ->- wR
    |            |
    |            ^
    |            p


If an MPO describes a sum of local terms (e.g. most Hamiltonians),
some bond indices correspond to 'only identities to the left/right'.
We store these indices in `IdL` and `IdR` (if there are such indices).

Similar as for the MPS, a bond index ``i`` is *left* of site `i`,
i.e. between sites ``i-1`` and ``i``.
```
![alt text](image.png)

### MPS（Matrix Product State）数据结构

```
r"""This module contains a base class for a Matrix Product State (MPS).

An MPS looks roughly like this::

    |   -- B[0] -- B[1] -- B[2] -- ...
    |       |       |      |

We use the following label convention for the `B` (where arrows indicate `qconj`)::

    |  vL ->- B ->- vR
    |         |
    |         ^
    |         p

We store one 3-leg tensor `_B[i]` with labels ``'vL', 'vR', 'p'`` for each of the `L` sites
``0 <= i < L``.
Additionally, we store ``L+1`` singular value arrays `_S[ib]` on each bond ``0 <= ib <= L``,
independent of the boundary conditions.
``_S[ib]`` gives the singular values on the bond ``i-1, i``.
However, be aware that e.g. :attr:`~tenpy.networks.mps.MPS.chi` returns only the dimensions of the
:attr:`~tenpy.networks.mps.MPS.nontrivial_bonds` depending on the boundary conditions.

The matrices and singular values always represent a normalized state
(i.e. ``np.linalg.norm(psi._S[ib]) == 1`` up to roundoff errors),
but (for finite MPS) we keep track of the norm in :attr:`~tenpy.networks.mps.MPS.norm`
(which is respected by :meth:`~tenpy.networks.mps.MPS.overlap`, ...).
```

![alt text](image-1.png)

### Quimb 中的 MPS 数据结构

![alt text](image-17.png)

---

## 开发日志

### 放假前遇到的问题

#### Git 仓库克隆问题
1. **问题描述**：使用 SSH 协议克隆 tensorcircuit-ng 仓库时下载进度卡住，多次尝试无果
   
   ![alt text](758aabe06cf1bb864de97912ceb6f10.png)

2. **尝试的解决方案**：
   - 切换文件夹与代理
   - 尝试直接通过网页克隆
   - 进行网络诊断：`ping github.com` 和 `curl -v https://github.com`
   - 尝试强制使用 IPv4

3. **问题分析**：
   - 可能原因：GitHub 限速、临时服务波动或校园网限制 Git 端口
   - 最终解决：隔天多次尝试浅克隆时成功，判断为 GitHub 网络波动导致

#### 其他问题
- **Quimb 文档缺失**：无法从 quimb 文档中找到 MPO 类，无法了解其数据结构
- **依赖安装问题**：
  - 使用 pip 安装 tensornetwork、tenpy、quimb 等扩展时，quimb 扩展出现版本不适配错误
  - 采用 Anaconda 进行扩展管理，调试时间较长
- **代码推送问题**：
  - 本地修改推送到云端时出现网络问题，多次上传失败
  - 最终通过分时段尝试成功上传
  
  ![alt text](image-5.png)

### 2025年7月26日

#### 环境配置重建
**背景**：由于之前进行 `check_all.sh` 检查时缺少很多库，决定放弃本地 Python 解释器，改用 Anaconda 读取 `requirements` 文件夹中的配置文件。

1. **Python 版本调整**：
   - TensorFlow 包对 Python 版本有特定要求
   - 使用 Anaconda 重新安装 Python 3.10 版本

2. **Rust 工具链安装**：
   - 某些 Python 包编译需要 Rust 工具链支持
   - 使用 PowerShell 安装 Rust 以适配权限要求

3. **环境重建**：
   - 原因：部分已安装的包可能在 Python 3.13 ABI 下编译，需要强制重建环境
   - 现象：命令行下载速度较慢
   
   ![alt text](image-7.png)

4. **依赖包安装进度**：
   - ✅ `requirements.txt` 安装完成
     
     ![alt text](image-8.png)
   
   - ✅ `requirements-dev.txt` 安装完成
     
     ![alt text](image-9.png)
   
   - ✅ `requirements-types.txt` 安装完成
     
     ![alt text](image-10.png)
   
   - ❌ `qcs-sdk-python` 库安装卡住
     
     ![alt text](image-11.png)

5. **替代方案**：
   - 可通过检查 GitHub 项目中的 CI 配置查看所需包：`.github/workflows/nightly_release.yml`
   
   ![alt text](image-16.png)

### 2025年7月27日

#### 依赖安装优化

1. **跳过问题库**：
   - 尝试跳过 `qcs-sdk-python` 库的安装
   - 逐行安装 `requirements-extra.txt` 中的其他库

   ```powershell
   Get-Content requirements-extra.txt | ForEach-Object {
       if (-not $_.Contains("qcs-sdk-python")) {
           pip install $_
       }
   }
   ```

   ![alt text](image-12.png)
   ![alt text](image-13.png)
   ![alt text](image-14.png)
   ![alt text](image-15.png)

   **结果**：仍出现一些错误，暂时跳过继续进行检查

2. **特定库安装**：
   - 由于代码中需要 tenpy 和 quimb 库，单独安装这两个库（需要 VPN）
   - `conda install -c conda-forge tenpy`
   - `conda install -c conda-forge quimb`

3. **代码检查**：
   - 运行 `check_all.sh` 检查
   - ✅ `black .` 可自动修正格式问题
   - ✅ `black check` 通过
   - ❌ `mypy check` 未通过，与老师交流后判断为 types 相关依赖安装不完全

4. **虚拟环境重设**：
   - 重新设置虚拟环境 `tc1`
   - ✅ `requirements.txt` 成功导入，无错误信息
     
     ![alt text](image-18.png)
   
   - ❌ `requirements-extra.txt` 在 `Building wheel for qcs-sdk-python` 时卡住
     
     ![alt text](image-19.png)

## 2025.7.28

- 再次尝试逐行安装`requirements-extra.txt`
    - 是`cirq → cirq-rigetti → pyquil → qcs-sdk-python`关联出现安装问题，询问老师去掉`cirq-rigetti`应该没有问题

- `pip install -r requirements/requirements-dev.txt`成功安装，无红字
    ![alt text](image-20.png)
- `pip install -r requirements/requirements-types.txt`成功安装，无红字
    ![alt text](image-21.png)
- `pip install requests`成功安装
    ![alt text](image-22.png)
- `pip install quimb`，`pip install physics-tenpy`成功安装
    ![alt text](image-24.png)

6. **Quimb 版本测试**：
   从 1.11 回退测试多个版本：
   - 1.0 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8
   
   ![alt text](image-29.png)
   ![alt text](image-30.png)
   ![alt text](image-31.png)
   ![alt text](image-32.png)
   ![alt text](image-33.png)
   ![alt text](image-34.png)

---

## 常用命令参考

### Conda 命令
- **激活虚拟环境**：`conda activate <env_name>`
- **清理缓存**：`conda clean --all -y`
- **按顺序安装依赖**：
  ```bash
  pip install --no-cache-dir -r requirements/requirements.txt
  pip install --no-cache-dir -r requirements/requirements-extra.txt
  pip install --no-cache-dir -r requirements/requirements-dev.txt
  pip install --no-cache-dir -r requirements/requirements-types.txt
  pip install requests
  ```

### Git Bash 命令
- **项目检查**：`./check_all.sh`
- **GitHub 连接测试**：`ssh -T git@github.com`