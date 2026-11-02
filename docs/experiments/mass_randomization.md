# 负载敏感性定位与仅质量随机化实验

日期：2026-11-02。数据来源：使用者本机提供的 nominal / added_mass 三个种子的 summary、逐回合 CSV 汇总与实际基座质量。仅质量随机化组已由使用者完成 1000 轮训练与 120 回合评估，观察到负载改善及低摩擦退化，见下方结果。

## 已观察的失败分类

|场景/种子|达到目标|超时|跌落或基座接触判定|基座质量 kg|失败回合平均时长 s|
|---|---:|---:|---:|---:|---:|
|nominal / 0|4|4|0|6.9210004806518555|20.02|
|nominal / 1|6|2|0|6.9210004806518555|20.02|
|nominal / 2|7|1|0|6.9210004806518555|20.02|
|added_mass / 0|1|1|6|8.921000480651855|9.934|
|added_mass / 1|2|2|4|8.921000480651855|13.15|
|added_mass / 2|1|2|5|8.921000480651855|14.494|

- nominal：17/24 达标，7 次失败均为超时。
- added_mass：4/24 达标，15 次跌落/基座接触判定、5 次超时。
- 记录的基座质量差确为 2kg。这里是基座质量，不是整机总质量。
- 负载场景主要新增了稳定性失败判定，而非只有速度不足。仍需按需回放失败回合、核对接触状态，不能仅凭汇总确定底层控制根因。
- nominal 失败时长实际报告为 20.02s，说明不能把名义 20s 当成精确连续时间边界。需进一步结合 metadata 中 dt 和 valid_steps 核对离散步长计数；本次不改变评估终止规则，以保持原模型与变体的评估实现一致。

## 变体与实验控制

新增参数：--qa-mass-rand。

- 在各环境创建时，对基座采样 [0, 2]kg 额外质量；不是在每个时间步或每次 reset 都改变质量。
- 相比现有平地基线，训练配置只启用基座质量随机化并设置上述范围。
- 保持原基线的观测噪声、摩擦随机化、推扰关闭状态；控制参数、奖励、学习率等沿用原配置。
- --qa-mass-rand 与原来的 --qa-domain-rand 互斥，避免把多种扰动混成一个实验。
- qa_training.json 记录 randomization_profile=mass_only；domain_randomization=true 表示存在随机化，不代表同时开启全部扰动。
- 预算与原基线一致：256 环境、1000 轮、训练 seed=42，从头训练。这是一组训练种子的探索实验，不能替代多训练种子的稳定结论。
- 原有 nominal、full 模式配置保持；本地 CPU 测试增加了配置隔离与互斥验证，总计 18 项通过。使用者已完成 GPU 训练/评估，结果见后文。

## 训练命令

在已配置库路径的 go2-test 环境中执行：

```bash
cd ~/robot_ws/go2-sim-test-bench
git pull --ff-only
mkdir -p reports/mass_only_training
set -o pipefail

python -u -m robot_test.train \
  --task=go2 \
  --qa-terrain plane \
  --qa-mass-rand \
  --num_envs=256 \
  --headless \
  --max_iterations=1000 \
  --seed=42 \
  --experiment_name=qa_plane_mass_rand \
  --run_name=seed42 \
  2>&1 | tee reports/mass_only_training/train.log
```

若使用新终端，需要先激活 go2-test，并按 README 设置当前环境的动态库路径。

## 回归协议与结果

训练后取得实际 model_1000.pt 路径，再用 robot_test.matrix 执行与原基线相同的 plane、20s / 10m、种子 0/1/2、每批 8 环境、五场景，共 120 回合。输出使用新目录，保留已有基线数据。

对比至少包括：
1. added_mass 达标数是否比原来的 4/24 提高。
2. added_mass 的 15 次跌落/基座接触判定是否减少。
3. nominal 达标数是否比原来的 17/24 退化，以及其他场景是否出现退化。
4. 各种子、时限、模型 SHA256 与实际质量记录是否完整。

不改变目标、时限或失败判定来抬高分数。质量随机化效果可能无提升甚至退化，实验结果以实际输出为准。

当前变体已完成：added_mass 13/24（基线 4/24），nominal 23/24（基线 17/24）；robot_friction 13/24（基线 22/24），存在退化。新模型失败类型尚未提供，不能据此断言跌落次数已减少。

完整数据与边界说明：[质量随机化回归对比](../results/mass_randomization_comparison.md)。

