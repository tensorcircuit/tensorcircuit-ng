- 关于tenpy_transform.md的修正
    - 遍历 `tenpy_obj.A` 获取张量列表
        - 实际为 `tenpy_obj._B`

# 新添加环境配置

- `tensorcircuit-ng/tensorcircuit/quantum.py`
![alt text](image-6.png)
```
from tenpy.networks import MPO, MPS, Site
import tenpy.linalg.np_conserved as npc
import tensornetwork as tn
from quimb.tensor import MatrixProductOperator
```
- 已在新建文件`tensorcircuit-ng/requirements/requirements-hyx.txt`中添加库依赖

# 数据结构分析

### TeNPy中`MPO`数据结构

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

### `MPS` 数据结构

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

### quimb中MPS数据结构

![alt text](image-17.png)

# 按日期记录

## 放假前
- 通过用ssh协议clone tensorcurcuit-ng仓库到本地时下载进度卡住，多次尝试无果
    - ![alt text](758aabe06cf1bb864de97912ceb6f10.png)
    - 通过切换文件夹与代理、尝试直接通过网页clone均无果，进行`ping github.com`与`curl -v https://github.com`网络诊断并尝试强制使用IPv4，仍然无法传输
    - 判断为 GitHub 限速或临时服务波动，或校园网被限速拦截 Git 端口
    - 隔天多次尝试浅克隆时成功下载到本地，应为 GitHub 本身网络波动所导致的传输失败

- 无法从quimb文档中找到class MPO，无法了解其数据结构

- 使用pip安装tensornetwork、tenpy、quimb等扩展时，quimb扩展安装时出现版本不适配报错
    - 下载anaconda对扩展进行管理，调试了很长时间

- 将本地修改push到云端时同样出现网络问题，多次上传失败，判断为同样是 GitHub 网络波动导致
    - ![alt text](image-5.png)
    - 分时段多次进行尝试，上传成功

## 2025.7.26

- 由于之前进行`check_all.sh`检查时缺少很多库，现抛弃使用本地 python 解释器，改用 ananconda 读取`requirements`文件夹中的`requirements.txt`等文件配置环境
    - tensorflow 包对 python 版本有要求，使用 anaconda 重装 `3.10` 版本 python
        - 需要安装 Rust 工具链以支持部分 python 包的编译
            - 使用 powershell 安装 Rust ，以适配 Rust 权限
        - 某些已安装的包可能在 Python 3.13 的 ABI 下编译的，需要强制重建环境
        ![alt text](image-7.png)
        （不知为何命令行下载速度很慢）
        - `requirements.txt` 库下载完成
        ![alt text](image-8.png)
        - `requirements-dev.txt` 库下载完成
        ![alt text](image-9.png)
        - `requirements-types.txt` 库下载完成
        ![alt text](image-10.png)
        - 卡在了`qcs-sdk-python`库安装的部分，尝试先独立安装该库
        独立安装库仍然会被卡住
        ![alt text](image-11.png)
        
        
    - 可通过检查Github项目中CI查看需要下载哪些包`.github/workflows/nightly_release.yml`
    ![alt text](image-16.png)

## 2025.7.27

- 尝试跳过`qcs-sdk-python`库的安装
    - 进行逐行安装`requirements-extra.txt`文件中的库

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
    还是出现了一些报错，暂且略过，重新进行`check_all.sh`检查

- 自己的代码中有tenpy与quimb库需求，单独下载该二库(需要vpn)
    - `conda install -c conda-forge tenpy`
    - `conda install -c conda-forge quimb`

- 进行`check_all.sh`检查
    - 可使用`black .`自动修正格式问题
    - `black check`通过
    - `mypy check`不通过，与老师交流判断为types相关依赖安装不完全，重新设置虚拟环境

- 重设置虚拟环境`tc1`
    - `requirements.txt`成功导入，无红字
    ![alt text](image-18.png)
    - `requirements-extra.txt`在中途`Building wheel for qcs-sdk-python`的时候卡住，与上相同
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

- 重新使用`tc`虚拟环境进行`check_all.sh`检查，mypy检查出现以下红字（无`quantum.py`）：
    ![alt text](image-23.png)

- 使用`tc1`虚拟环境进行`check_all.sh`检查，mypy 检查出现以下红字（无`quantum.py`）：
    ![alt text](image-25.png)

- 跳过 mypy 检查，pylint检查通过：
    ![alt text](image-26.png)


- 找到了报错原因，在 Git bash 中没有使用conda虚拟环境，而是直接用的本地python解释器
    - 使用虚拟环境后，前三个检查顺利通过
    ![alt text](image-27.png)

    - pytest 检查如下（644 passed）：
    ![alt text](032a743d0ec241502599f0ea943a21a.png)
    - 将numpy从1.24升级到1.25（支持exceptions，647 passed）
    （ numpy 不能大于一定版本，否则 mypy check 无法通过）
    ![alt text](image-28.png)
    - quimb从1.11回退到1.0
    ![alt text](image-29.png)
    quimb --version=1.4
    ![alt text](image-30.png)
    --version=1.5
    ![alt text](image-31.png)
    --version=1.6
    ![alt text](image-32.png)
    --version=1.7
    ![alt text](image-33.png)
    --version=1.8
    ![alt text](image-34.png)


# 常用代码

## Conda

- 激活虚拟环境：`conda activate <env_name>`
- 清理内存`conda clean --all -y`
- 按顺序下载：
`pip install --no-cache-dir -r requirements/requirements.txt`
`pip install --no-cache-dir -r requirements/requirements-extra.txt`
`pip install --no-cache-dir -r requirements/requirements-dev.txt`
`pip install --no-cache-dir -r requirements/requirements-types.txt`
`pip install requests`

## Git Bash

- 检查：`./check_all.sh`
- 连接Github：`ssh -T git@github.com`