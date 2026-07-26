# TensorCircuit-NG BF16 Phase 0 完整研究报告

日期：2026-07-27

报告快照：`03b8b45f16c06cf550481241a7f380e3e55265a0`

范围：BF16/planar-complex、grouped、CUTLASS 与 region-fusion 在目标 GPU 上的能力、显存、性能、数值和证据完整性评估

## 1. 执行摘要

Phase 0 得到的是一个有价值但尚未闭环的研究结果：**BF16 与 region fusion 的显存/性能杠杆是真实的，但截至本报告，没有一条路线同时完成能力和数值门控，因此 Phase 1 仍未获授权。**

当前结论如下：

| 项目 | 当前结论 |
|---|---|
| Phase 0 | **INCONCLUSIVE** |
| Phase 1 | **NOT_AUTHORIZED** |
| 已确认可行路线 | **0 条** |
| `planar` | 能力 PASS；数值覆盖不完整，**UNKNOWN** |
| `grouped` | 目标异构 grouped API 不支持，**NOT_VIABLE** |
| `region_fused/direct` | 实测节省 1 GiB，但 v5 精度 18/18 失败，**NOT_VIABLE** |
| `cutlass_4m_single`（SM80 fallback） | 能力 PASS；完整数值矩阵未测，**UNKNOWN** |

这不是“BF16 没有价值”的结论。相反，Phase 0 已经确认三件关键事实：

1. 生产 XLA 图中确实存在 128–512 MiB 级别的物化中间量，研究对象不是被编译器消除的虚假杠杆。
2. planar BF16 和 CUTLASS SM80 fallback 均展示了真实运行能力与明显性能潜力。
3. full-anchor region fusion 确实能避开两个各 512 MiB 的中间量，将测得的 allocator peak 从 1696 MiB 降至 672 MiB。

但 direct region kernel 使用顺序 FP32 累加和高倍 producer 重算。它虽然通过所有 18 个单元的全局相对 L2 门控，却在所有 18 个单元中失败于局部缩放最大误差门控；这说明全局 L2 会把 67,108,864 个输出元素中的局部误差稀释掉。该路线不能再声称 VIABLE，也不应在看到结果后放宽阈值。

## 2. 报告口径与权威状态

仓库中存在两个时间层次的状态，必须分开理解。

### 2.1 已生成的旧 canonical 汇总

当前 `gonogo.json` 和 `manifest.json` 仍来自 v5 18-cell 结果之前的生成链：

- `manifest.json.generated_at = 2026-07-26T12:10:30Z`
- aggregation source：`976c7892fa575758f14ce63677aa733b97961ac4`
- measurement source：`205899678c0de72e9ff180ab357a973bf7e1112e`
- 其中 `region_fused`、`REGION_PROTOTYPE` 和 `C2_REGION_KERNEL_FEASIBILITY` 仍为 `UNKNOWN`
- `review_subject.json` 仍指向更早的 subject `bc6294a76bbe20f8ebe6bae08fa9434a8ece86ff`

因此，这些文件可以说明旧生成链的状态，但**不能覆盖** 2026-07-27 已提交的 v5 GPU 结果。

### 2.2 最新研究证据

最新结果提交为：

```text
03b8b45f16c06cf550481241a7f380e3e55265a0
results(phase0): record region-fused v5 18-cell GPU accuracy failure
```

该提交包含完整 18-cell CSV、聚合 JSON、运行日志、run context 和专门研究报告。按当前门控代码重新生成下游链后，预期变化为：

| Criterion | 当前旧汇总 | 最新证据应导出的状态 |
|---|---|---|
| `C1` | PASS | PASS |
| `C2_REGION_KERNEL_FEASIBILITY` | UNKNOWN | **FAIL**（v5 accuracy FAIL） |
| `C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK` | FAIL | FAIL |
| `C2_JOINT_EXECUTABLE_LEVERAGE` | UNKNOWN | UNKNOWN |
| `C2_CANONICAL` | UNKNOWN | **FAIL**（region FAIL 会确定性下沉） |
| `C3_PLANAR_CORE` | PASS | PASS |
| `C3_PLANAR_FULL_MATRIX` | PASS | PASS |
| `C3_GROUPED` | NOT_SUPPORTED | NOT_SUPPORTED |
| `CUTLASS_SM120_4M` | NOT_SUPPORTED | NOT_SUPPORTED |
| `CUTLASS_SM80_FALLBACK_CAPABILITY` | PASS | PASS |
| `REGION_PROTOTYPE` | UNKNOWN | **FAIL** |
| `NUMERICAL` | UNKNOWN | UNKNOWN（planar/CUTLASS 仍未闭环） |

上表右列是对现有门控规则和最新原始证据的研究解释，不冒充尚未执行的 canonical 再生成结果。即使 region 相关项从 UNKNOWN 变为 FAIL，Phase 0 仍是 INCONCLUSIVE，因为 `C2_JOINT_EXECUTABLE_LEVERAGE` 与总体 `NUMERICAL` 仍未确定。

## 3. 研究目标与判定模型

Phase 0 的目标不是完成生产实现，而是回答以下问题：

1. 生产执行图是否真的物化了足够大的中间量？
2. BF16/planar、grouped、CUTLASS 或 region fusion 是否能在目标硬件上执行？
3. 候选实现是否产生可测量的显存或速度收益？
4. 候选实现是否满足预先冻结的数值策略？
5. 是否至少存在一条 capability 与 numerical 均通过、且证据绑定完整的 VIABLE 路线？

路线判定采用能力与数值的双条件：

```text
capability = OK 且 numerical = PASS     -> VIABLE
任一侧为 NOT_OK                        -> NOT_VIABLE
否则                                   -> UNKNOWN
```

Phase 状态使用更严格的真值表：

```text
全部 12 个 canonical criteria 均已确定 + 至少一条 VIABLE -> GO_TO_PHASE1
全部 criteria 已确定 + 没有 VIABLE                    -> NO_GO
任一 required criterion 为 UNKNOWN/NOT_RUN            -> NOT_AUTHORIZED
```

注意：FAIL 是“已确定的负结果”，不会单独导致 Phase 0 不完整；UNKNOWN 才是阻止完成判定的状态。

## 4. 测试环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | 12.0 (`sm_120`) |
| SM 数量 | 46 |
| 显存 | 约 12.82 GB |
| Driver | 592.47 |
| CUDA runtime | 12.8 |
| PyTorch | 2.11.0+cu128 |
| JAX / jaxlib | 0.6.2 / 0.6.2 |
| CuPy | 14.1.1 (`cupy-cuda12x`) |
| cuBLAS | 12.8.4.1 |
| NumPy | 2.2.6 |
| TensorCircuit-NG | 1.7.0 |
| TF32 matmul | 禁用 |

Tracked artifacts 中的机器路径经过脱敏；提交、源文件哈希、策略哈希和依赖版本构成主要复现指纹。

## 5. C1：物化中间量验证

C1 的目的，是避免把“理论上很大、实际上被 XLA 融掉”的 tensor 当成内存优化对象。两组规模均通过全部六项条件：动态参数、HLO 存在物化 buffer、buffer 至少达到半个 state、未被 XLA 消除、可执行、三次峰值重复稳定。

| Case | default peak | no-fusion peak | full state | 最大物化 HLO buffer | 结论 |
|---|---:|---:|---:|---:|---|
| `n24_d10` | 1056.17 MiB | 1056.42 MiB | 128 MiB | 512 MiB | PASS |
| `n22_d10` | 256.08 MiB | 256.31 MiB | 32 MiB | 128 MiB | PASS |

两种 fusion 设置的峰值几乎相同，说明简单切换 XLA fusion 并不能自动释放这个显存杠杆。C1 的可靠结论是：**大中间量真实存在且稳定可复现，值得继续优化。**

## 6. C2：显存杠杆、single-anchor 与 joint 证据

### 6.1 生产图的峰值结构

在 `n24_d10_default` 的 XLA buffer assignment 中，基准峰值约为 1.056 GiB。anchor 为：

```text
P = A @ B
T = transform(P)
E = D @ T
```

在峰值时刻，512 MiB 的 `T` 与 512 MiB 的 `E` 同时存活。`P` 与后续输出复用了同一物理 allocation，不在峰值 live set 中。因此，在“其余程序调度保持不变”的反事实模型里，仅消除 anchor 的 `P/T` 会把峰值移到别处，而不是消除整个链的结构峰值。

### 6.2 single-anchor 结论

| 项目 | 结果 |
|---|---:|
| memory threshold | 268,435,456 B（256 MiB） |
| single-anchor peak reduction | 31,872 B |
| `C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK` | **FAIL** |

这意味着单个 production anchor patch 不是全程序峰值解决方案。

### 6.3 joint 模型结论

五个候选窗口的反事实 joint model 给出最大峰值下降 704,736,544 B，理论上超过 256 MiB 门槛。但该模型没有计入 fused workspace、重算和真实执行调度，也没有对应的 joint executable。

因此：

```text
C2_JOINT_EXECUTABLE_LEVERAGE = UNKNOWN
```

模型上界不能替代真实执行。要关闭该 UNKNOWN，必须有一个能运行的 joint/whole-chain 实现；测得低于阈值也可以形成确定的 FAIL。

### 6.4 full-anchor region kernel 的局部显存结果

独立 full-anchor prototype 则证明局部 region-fusion 杠杆是真实的：

| 路径 | allocator peak |
|---|---:|
| materialized `P -> T -> E` | 1,778,384,896 B（1696 MiB） |
| fused direct | 704,643,072 B（672 MiB） |
| 实测下降 | **1,073,741,824 B（1024 MiB）** |

kernel 没有分配完整 `P` 或 `T`，分别避开 512 MiB；workspace/allocator 范围为同一 full-anchor scope。该结果说明 **region fusion 是真实的局部内存杠杆**，但不等同于整个 contraction chain 已经降低 1 GiB，也不等同于数值路线可行。

## 7. C3 与候选执行路线

### 7.1 cuBLASLt planar

planar 路线将 complex GEMM 展开为实数 BF16 运算。能力矩阵覆盖：

```text
8 shapes x 2 output dtypes x 4 workspace caps x 2 operation modes = 128 cells
```

其中 120 个单元 `ok`，8 个 `no-algo`；四个 `min(M,N,K) >= 16` 的 real-GEMM 形状全部通过能力 quorum，因此：

```text
C3_PLANAR_CORE = PASS
C3_PLANAR_FULL_MATRIX = PASS
```

四个 real-GEMM 形状相对 c64 kernel-only 的公平速度比分别约为 7.77×、4.75×、4.60× 和 3.31×。极瘦的小尺寸只是诊断项，其中部分比 c64 慢，不能用于宣称全形状统一加速。

能力 PASS 不代表数值已经闭环。现有 numerical aggregate 对 planar 期望 144 个单元，只识别到 96 个当前有效单元；缺少 `cancellation_v2` 的：

```text
8 shapes x 2 output modes x 3 seeds = 48 cells
```

同时存在 48 个旧 `cancellation_legacy_v1` extra rows，它们不能替代 v2。因此 `planar = UNKNOWN`。

### 7.2 cuBLASLt grouped

同形状 batched 路径 64/64 能力单元通过，但目标 contraction 需要异构 grouped GEMM。测得的 cuBLASLt 头文件中没有 grouped-3GEMM descriptor API；legacy grouped batched API 又没有 planar-complex 所需的 `PLANE_OFFSET` 布局。

因此：

```text
C3_GROUPED = NOT_SUPPORTED
grouped = NOT_VIABLE
```

这里的 NOT_SUPPORTED 是目标 API/布局不匹配，不是否定所有 batched GEMM。

### 7.3 CUTLASS

原生 SM120 BF16 4M kernel 编译失败。CUTLASS builder 报告当前 SM120 TMA warp-specialized 路径只支持 F8/F6/F4，且找不到匹配的 BF16 MMA：

```text
CUTLASS_SM120_4M = NOT_SUPPORTED
```

独立的 SM80 fallback 可以在目标 GPU 上编译和运行：

| 测试 | 结果 |
|---|---:|
| 单个 4M GEMM，3 seeds max_rel | 6.5472e-5 |
| fallback kernel-only latency | 3.2167 ms |
| c64 baseline latency | 16.9107 ms |
| kernel-only speed ratio | **5.257×** |
| workspace | 0 B |

故 `CUTLASS_SM80_FALLBACK_CAPABILITY = PASS`。但 canonical numerical matrix 要求 baseline、mixed-scale、cancellation 三个 profile 共 9 cells，目前有效测量为 0/9；旧 capability correctness 不能替代该矩阵。因此 `cutlass_4m_single = UNKNOWN`。

异构 grouped fallback 也完成了 8 个代表形状的 seed-0 correctness，但约 4.063 ms 对 2.234 ms c64 baseline，当前实现没有性能优势。

## 8. region_fused 的实现、性能与数值结论

### 8.1 frozen full-anchor contract

```text
P = A[4096,1024] @ B[1024,16384] -> c64[4096,16384]
T = transform(P)                 -> c64[64,1048576]
E = D[64,64] @ T                 -> c64[64,1048576]
```

`P`、`T` 与 `E` 各为 512 MiB。direct kernel 通过现场重算 producer 避免完整 `P/T`，producer recompute factor 为 64，估算重算量为 8,796,093,022,208 FLOPs。

### 8.2 性能探索

| 变体 | 最佳 kernel-only latency | 相对 direct |
|---|---:|---:|
| direct | 20,210.1 ms | 1.0× |
| tiled | 3,007.6 ms | 6.7× |
| persistent | 1,148.5 ms | 17.6× |

最新 direct 完整测量为 20,412.5 ms，而 materialized 路径为 101.27 ms，约慢 202×。tiled/persistent 明显改善调度，但仍使用与 direct 相同的顺序 FP32 producer/consumer 累加结构。它们旧的 `rel_l2` 诊断值不能继承为 v5 局部精度 PASS。

### 8.3 v5 双门控策略

策略在重新测量前冻结：

```text
s = sqrt(sum_i |r_i|^2 / n)
global_rel_l2 = ||o-r||_2 / ||r||_2
local_scaled_max = max_i |o_i-r_i| / max(|r_i|, alpha*s)

alpha = 1e-3
global_rel_l2 < 1e-4
local_scaled_max < 1e-3
```

不等式严格，小于阈值才通过。策略 commit 为 `30a0048b09f6f7f58d9fa72ea8eacbd161ca382a`，策略文件 SHA-256 为 `3ecfa370409e2397319276b8aa1b64bf19a816b2e8e0fb478b51569bf383ced1`。

测量矩阵为三个 profile × 六个 seeds：

- calibration：`0, 1, 2`
- holdout：`1598166685, 542109305, 1463850203`
- profile：`baseline_v1`、`mixed_scale_v1`、`cancellation_v2`

18/18 cells 完成，无 OOM、timeout、基础设施故障或重试；全部 finite，policy identity 和覆盖均完整。

### 8.4 v5 测量结果

| Profile | Cells | 最大 global rel-L2 | local-scaled-max 范围 | argmax 处 `|ref|` | 结论 |
|---|---:|---:|---:|---:|---|
| baseline | 6 | 8.5113e-7 | 1.6523e-3 – 2.0868e-3 | 0.572 – 0.840 | FAIL |
| mixed-scale | 6 | 7.4816e-7 | 1.4297e-3 – 2.3253e-3 | 1.25e5 – 3.39e5 | FAIL |
| cancellation | 6 | 5.8129e-5 | 1.1744e-1 – 1.4666e-1 | 2.12e-4 – 5.45e-4 | FAIL |

所有 18 cells 均：

- 通过 `global_rel_l2 < 1e-4`
- 失败于 `local_scaled_max < 1e-3`
- 原因均为 `FAIL_LOCAL_SCALED_MAX`

最差单元为 `cancellation:cancellation_v2:seed=542109305`：

```text
global_rel_l2 = 5.8128938069216e-5       PASS
local_scaled_max = 0.14666077359851473  FAIL
argmax |ref| = 0.0004882161863755533
nan_inf = false
```

低幅值输出确实放大了 cancellation profile 的局部误差，但它不是全部原因：baseline 和 mixed-scale 在正常或极高幅值元素上也稳定超过 1e-3。原先“只过滤极小输出即可解决”的假设被实验否定。

因此：

```text
region_fused/direct accuracy = FAIL
region_fused/direct = NOT_VIABLE
region-fusion memory leverage = CONFIRMED
```

## 9. 为什么当前仍有 UNKNOWN

UNKNOWN 表示证据不完整或没有可执行证明，不表示结果接近 PASS。

| UNKNOWN | 具体原因 | 最小关闭动作 |
|---|---|---|
| planar numerical | 缺少 48 个 `cancellation_v2` cells | 运行 48 cells；若旧绑定失效则重跑完整 144 cells |
| CUTLASS fallback numerical | 缺少 9 个 profile/seed cells | 运行固定 9-cell 矩阵 |
| `C2_JOINT_EXECUTABLE_LEVERAGE` | 只有 704.7 MB 的模型上界，没有 executable | 构建并测量 joint/whole-chain attempt |
| 总体 `NUMERICAL` | 上述路线仍未完全确定 | 重新聚合完整 per-route 结果 |
| 旧 canonical 链 | 尚未吸收 v5 18-cell FAIL | 依赖顺序再生成 C2/gonogo/manifest/closeout |

当前 UNKNOWN 是门控按设计拒绝猜测的结果。特别是 joint model 即使超过门槛，也不能在没有运行实现时变成 PASS。

## 10. Phase 1 是否可以启动

**现在不可以。** 当前 Phase 0 是 INCONCLUSIVE，真值表强制给出 `NOT_AUTHORIZED`。

即使下一步 planar 通过并成为 VIABLE，也只能证明“已有候选路线”。要得到 `GO_TO_PHASE1`，还需要关闭所有 required criterion 的 UNKNOWN，尤其是 joint executable leverage 和总体 numerical aggregate，然后重新生成、绑定并审阅完整 canonical 链。

如果所有 UNKNOWN 均关闭且至少一条路线 VIABLE，则进入 Phase 1；如果全部关闭但没有路线 VIABLE，则结论应为确定的 `NO_GO`，而不是继续保持 INCONCLUSIVE。

## 11. 推荐后续路线

按成本和成功概率，建议顺序如下。

### A. 优先闭环 planar

planar 已经通过能力门控，且在四个 real-GEMM 形状上有 3.31×–7.77× 的 kernel-only 潜力。首先补齐 48 个 `cancellation_v2` cells；这是最短、最便宜、最可能产生首条 VIABLE 路线的工作。

### B. 必要时闭环 CUTLASS SM80 fallback

若 planar 失败、仍不确定，或需要第二条候选路线，再完成 9-cell CUTLASS 数值矩阵。不要把 native SM120 的 NOT_SUPPORTED 与 fallback 能力混为一谈。

### C. 若仍追求 1 GiB 显存杠杆，开发 streamed/blockwise region fusion

新实现应使用 GEMM 风格的分块/树归约计算 producer tile，再流式送入 consumer；不得沿用 direct kernel 的逐输出顺序 producer 重算。新 variant 必须使用新身份，重新预冻结策略、实现、seeds、workspace 与峰值范围。建议继续使用相同双门控常量，不放宽 v5 阈值。

### D. 最后关闭 canonical UNKNOWN

无论 A/B/C 哪条路线成功，都必须：

1. 形成 executable joint attempt，确定 `C2_JOINT_EXECUTABLE_LEVERAGE`；
2. 按依赖顺序再生成 numerical、region、C2、gonogo、manifest 和 closeout；
3. 创建指向干净结果提交的新 `review_subject.json`；
4. 由 Reviewer B 审查结果绑定和最终 verdict。

后续恢复路线的执行合同正在独立起草，本阶段成果 PR 有意不包含该草稿；本节仅记录已经由当前证据支持的建议顺序。

## 12. 审计与复现限制

### 12.1 v5 measurement source 的轻微偏差

原 freeze candidate 为 `fc8b2c1d522861beaa849d808b0fc8a9c6dab873`。测量前 `region_proto.py` 增加了 7 行 profile version-token 映射，随后 freeze commit amend 为 `09e69b9fe9542879a13f74fcca3f6e51a53e8253`。该差异只改变 summary cell-key 文本，不改变输入、CUDA kernel、materialized oracle 或双门控指标。

这不影响本报告的研究结论，但对严格审计而言仍是 provenance deviation；最终 closeout 应使用一致的预冻结 commit 重新绑定。

### 12.2 下游制品未再生成

`c2_judgment.json`、`numerical_validation.json`、`gonogo.json`、`manifest.json`、`closeout_facts.json` 和 `review_subject.json` 尚未基于结果提交 `03b8b45f` 全链再生成。因此它们内部仍可见旧 PASS/UNKNOWN 或旧 hash。报告没有手工修改这些 derived artifacts，以免伪造 canonical 状态。

### 12.3 测试口径

仓库保存的 `test_report.json` 记录命令 `python -m pytest results/_phase0/ -m 'not gpu'` 退出码为 0。v5 实现阶段另有门控与 mutation 测试记录；本报告生成过程中没有重新运行 GPU 测量，也没有把历史测试报告解释为最新 result commit 的独立审阅接受。

### 12.4 工作树

本报告生成前 tracked worktree 无修改；仓库存在预先已有的 untracked scratch probes、XLA dumps 和 handoff 文件。它们未被删除，也未被纳入结果提交。当前 `review_subject.json` 的 `dirty_worktree=false` 只描述其历史 subject，不描述本报告生成时的整个工作目录。

### 12.5 非阻塞文本与字段债务

- v5 策略文件标题仍写有 `DRAFT v5`，而 freeze manifest 和提交记录已绑定 Reviewer B 的 `POLICY_ACCEPTED`。最终 closeout 应统一文字状态。
- `region_prototype.json.peak_measurement_method` 仍使用旧名称 `raw_allocation_size_delta`，而实际 runtime 字段为 `cuda_allocator_highwatermark`。当前 artifact 另有 `peak_evidence_class=MEASURED`、同范围峰值和明确的 runtime method，故这属于命名债务，不改变本次 1696 MiB 对 672 MiB 的研究测量；再生成时应统一名称。
- `closeout_facts.json` 中的旧 `REGION_PROTOTYPE=PASS` / `NUMERICAL=FAIL` 也不应继续作为当前结论；它必须与其余下游制品一起由最新输入重建。

## 13. 结论

Phase 0 的总结果应表述为：

> **显存杠杆和若干 BF16 性能路径已经被实测确认，但尚未得到一条证据闭环的 VIABLE 路线。direct region fusion 因结构性局部精度失败而淘汰；planar 与 CUTLASS fallback 仍有现实可行性，但必须补齐数值矩阵。Phase 0 保持 INCONCLUSIVE，Phase 1 保持 NOT_AUTHORIZED。**

这份结论既不是乐观地把内存 PASS 等同于路线可行，也不是悲观地否定 BF16。它把已经确认的收益、已经确定的失败和仍需测量的未知严格区分开来。

## 14. 核心制品索引

| 内容 | 文件 |
|---|---|
| C1 判定 | `results/phase0/c1_judgment.json` |
| C1/C2 anchor 映射 | `results/phase0/c1_c2_edge_map.json` |
| C2 peak frontier | `results/phase0/c2_peak_frontier.json` |
| C2 当前旧判定 | `results/phase0/c2_judgment.json` |
| planar 能力 | `results/phase0/cublaslt_planar_capability.json` |
| grouped 能力 | `results/phase0/cublaslt_grouped_capability.json` |
| CUTLASS 能力 | `results/phase0/cutlass_sm120_4m.json` |
| 当前 numerical aggregate | `results/phase0/numerical_validation.json` |
| v5 策略 | `docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md` |
| v5 freeze manifest | `results/phase0/policy_freeze_manifest.json` |
| v5 region aggregate | `results/phase0/region_prototype.json` |
| v5 18-cell 原始表 | `results/phase0/region_prototype_accuracy.csv` |
| v5 运行日志 | `results/phase0/region_fused_v5_research_run.log` |
| v5 专项报告 | `results/phase0/region_fused_v5_research_report.md` |
| 当前旧 go/no-go | `results/phase0/gonogo.json` |
| 当前旧 manifest | `results/phase0/manifest.json` |
