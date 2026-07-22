# Phase 0 工具安装记录（日期：2026-07-22）

- **nsys**: unavailable。WSL2 `sudo` 需密码（无法非交互 `apt install`），conda-forge / nvidia channel 均无 `nsight-systems` 包。
  → Probe 3 退回 **lens 1（静态 HLO + buffer 计数）+ lens 2（`--xla_disable_hlo_passes=fusion` A/B peak 比）**。lens 3（nsys 时间线）跳过。
  **不阻塞 Phase 0**：lens 2 的 `peak_default vs peak_no_fusion` 比是 spec §4.2 的决定性信号，不依赖 nsys。
- **ncu**: missing。仅将来 **post-go/no-go** 的 Probe 1 libcublasLt 绑定需要（验 Tensor Core SASS）。
  → fallback 为 `torch.profiler` kernel 名 dump（现有 `_k3_pytorch_probe` 模式）。本计划不依赖。
- **Probe 3 nsys 依赖**: 否（用 lens 1+2）。
- **若日后 lens 1+2 不足**：请用户用其 sudo 执行 `sudo apt-get install -y nsight-systems-cli`（或 NVIDIA 独立包），再补 lens 3。
