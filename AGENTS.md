# Brush fork 维护指南

## 工作约定

- 以 `origin/main` 为 fork 基线，以 `upstream/main` 跟踪 ArthurBrussee/brush。同步上游与功能改动分别提交，保留已有 fork 行为。
- 按仓库 CI 使用 Rust 1.100.0、Cargo workspace 和锁定依赖。修改 manifest 后用 Cargo 更新 `Cargo.lock`。测试走实际加载、渲染和训练入口，断言能在恢复原缺陷时失败。
- 除非用户另有说明，禁止使用 Superpowers skills。沿用仓库流程，避免预防性改动和内部重复校验。新增功能默认关闭，默认路径须有回归对照。
- 主机承担方案、执行拆解、环境与依赖、故障排查和验收。使用执行协作时，给出已验证的输入、命令、改动边界和停止条件。

## Issue #3 漂浮 splat

实施基线 `6d73e74` 已纳入 [原仓六个提交](https://github.com/ArthurBrussee/brush/compare/1388f74...d43bd1e)。保留本地 `main` 作为同步基线，在维护分支实施。按以下顺序检查交付，每步保留独立的基线与功能开关。[fork issue #3](https://github.com/Milesssssss/brush/issues/3)

1. 补实已有 split 与 reset。保留 `k >= 0.5`、弱子体可剪枝以及满预算停止分裂的行为。修正训练测试相机，断言实际可见覆盖和参数更新。验证默认 3D filter 下弱子体的剪枝。在真实训练循环中检查 refine 先于 reset、有效 alpha 上限（默认 3D filter 计入）、Adam 状态、断点恢复和阶段截止。针状 splat 的贴合偏移单独做 A/B，比较前景细节和侧视／端视轮廓外 alpha 再决定是否采用。[fork #6](https://github.com/Milesssssss/brush/pull/6) · [fork #5](https://github.com/Milesssssss/brush/pull/5)
2. 接入 fork #4 的 `4115ef9`，验证 COLMAP TIFF、尺度估计与初始化 PLY、遮罩、无效深度和两个预览入口。度量模式继续用摄像机 Z 的逆深度误差。权重为零保留四通道 RGB 训练路径。用真实 `SplatTrainer::step` 验证遮罩外深度变化无影响、全零遮罩和非有限目标，以及旋转相机下有可预期的深度梯度和损失下降。验证五通道渲染输出的 RGBA 与普通路径一致。[fork #4](https://github.com/Milesssssss/brush/pull/4) · [原仓 #497](https://github.com/ArthurBrussee/brush/pull/497)
3. 复用 TIFF 与深度通道，加入可选相对逆深度模式。约定近处值大、无单位，样本采用遮罩加权的局部与全局归一化。相对值绕过度量尺度估计和单位变换，预测器保持离线。覆盖正仿射变换、深度符号反转、边缘 patch、空权重和真实训练更新。[DNGaussian 参考](https://github.com/Fictionarry/DNGaussian/blob/main/utils/loss_utils.py#L73-L83)
4. 加入可选的每训练图 3×4 RGB 仿射，从恒等初始化，正则约束，只修正前景 RGB。用数据集相对图像路径匹配评估和 checkpoint，保存参数及 Adam 状态。留出视图和自由相机使用标准颜色，PLY 仍可独立查看。覆盖遮罩和透明输入、视图顺序变化、断点重载，以及受控曝光变化的 A/B。借鉴 #483 的视图索引和评估匹配，只有 A/B 显示存在剩余空间性或非线性色偏才评估 bilateral grid 或 PPISP。[原仓 #483](https://github.com/ArthurBrussee/brush/pull/483)

复用已纳入的 #479 分裂与滤波、#562 checkpoint、#512 光栅化和 #565 macOS 原生 Metal。#353 的多视图剪枝待以上实验有数据后再评估。[#479](https://github.com/ArthurBrussee/brush/pull/479) · [#562](https://github.com/ArthurBrussee/brush/pull/562) · [#512](https://github.com/ArthurBrussee/brush/pull/512) · [#565](https://github.com/ArthurBrussee/brush/pull/565) · [#353](https://github.com/ArthurBrussee/brush/pull/353)

### 分裂偏移决定

单针渲染 A/B 中，紧凑偏移减少了侧视轮廓外 alpha，但把轮廓覆盖从 100% 降到约 86%–89%，并增加了轮廓内 alpha 误差。保留现有分裂偏移。可用 `cargo +1.100.0 test --locked -p brush-train --lib compare_needle_split_offsets -- --ignored --nocapture` 复现。

## 验收

- 运行仓库格式、Clippy、原生测试和 WASM 编译。WGPU 单元测试、实际 Metal 训练和浏览器／桌面预览各自记录证据。编译通过只记录为编译结果。
- 组合短跑覆盖实际 split、refine、reset 与 checkpoint 恢复。resume 测试对照相同批次或对照调度事件，现有 checkpoint 会重新启动数据加载顺序。
- 固定数据版本、相机、遮罩、初始 PLY、训练／留出划分、采样顺序、后端和训练预算。记录轮廓外 alpha、前景覆盖与细节、留出 RGB 误差，以及独立真值深度误差。用重复基线设定容差，再比较单项和组合。切换深度预览时检查有效区间、无效值、单位与缺失深度的行为。
