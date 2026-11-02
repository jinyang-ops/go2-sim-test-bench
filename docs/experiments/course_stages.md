# 比赛地图短程分段训练

目的：在[迁移退化对照](../results/competition_transfer_regression.md)之后，从已验证可运动的初始模型恢复，缩小下一轮训练覆盖跨度，尽早检查运动能力是否保留。

状态：分段入口已在实验端完成一次 100 轮 GPU 训练，但短程运动检查失败。仅缩小训练区间未能保留该测试下的运动能力；暂停扩大区间与追加训练，先执行[离线诊断](transfer_diagnostics.md)。以下命令保留为失败实验的配置记录，不作为当前恢复建议。

## 训练区间

`robot_test.train --qa-terrain competition --qa-course-stage <name>` 使用独立训练子类，记录 course_stage 到 qa_training.json。仅接受参考 12m×108m 地形布局，初始位置 XY 偏移必须为零。

|名称|出生原点 (x,y,z) m|段末世界 X m|覆盖|
|---|---|---:|---|
|flat|(2,6,0)|10|地图起始平坦区域，距坡面起点保留 2m|
|front|(8,6,0)|24|平地起步，经过 12–24m 坡面|

两种出生原点都在平地；机身高度仍由原配置 init_state.pos[2] 提供。原初速度随机化、命令采样、25s 最大回合长度、奖励、优化器配置及探索参数保留。下一轮不同时增加质量/摩擦/推扰等训练扰动。

原物理失败优先于段末达标。y≤0 或 y≥12 触发训练重置；到达段末且该帧无原终止条件时，记为人为截断，通过原 time_outs 接口进行价值引导，不添加达标奖励。该训练边界与正式评估是不同用途，评估协议不受此训练子类影响。

TensorBoard 的 stage_goal_fraction 是当次重置环境中的段末达标比例，不是完整评估成功率。原奖励日志仍按配置的最大回合时长归一化；提前截断会改变奖励汇总的解释，不能横比不同区间的 Mean reward 判断优劣。

## 第一轮：只训练 flat，100 次更新

从原始平地质量随机化 model_1000 初始化，不从退化的地图 model_2000 继续。使用新实验名，不覆盖历史目录。已激活仿真环境并设置动态库路径后：

```bash
cd ~/robot_ws/go2-sim-test-bench
git -c http.version=HTTP/1.1 pull --ff-only
# 确认拉取成功后再执行后续命令。
init_model=/absolute/path/to/qa_plane_mass_rand/model_1000.pt
mkdir -p reports/course_flat_recovery
set -o pipefail

python -u -m robot_test.train \
  --task=go2 --qa-terrain competition --qa-course-stage flat \
  --qa-init-model "$init_model" \
  --num_envs=128 --headless --max_iterations=100 --seed=42 \
  --experiment_name=qa_course_flat_recovery --run_name=seed42 \
  2>&1 | tee reports/course_flat_recovery/train.log
```

保留源权重、配置、初始化哈希与训练日志；model_100 表示这次新实验的更新数。model_0 是原训练循环首次更新后保存的文件，不能充当精确的未训练初始化对照。

## 检查后再扩展

先重测与已知对照相同的短程条件，用新目录保存：

```bash
candidate=/absolute/path/to/qa_course_flat_recovery/model_100.pt
python -m robot_test.evaluate \
  --task=go2 --headless --num_envs=1 --seed=0 \
  --qa-model "$candidate" --qa-output reports/course_flat_100_sanity_v2 \
  --qa-terrain competition --qa-case nominal --qa-heading-mode heading \
  --qa-speed 1.0 --qa-duration 5 --qa-goal-m 1 --qa-trace
```

单回合达标只作为最基本检查。随后将源模型与候选模型都放进同一矩阵：competition、nominal、heading、seeds 0/1/2、每个 seed 8 环境、时限 15s、前进目标 8m；分别使用全新输出目录并记录轨迹。建议命令：

```bash
python -m robot_test.matrix --model "$candidate" \
  --output reports/course_flat_100_matrix_v2 \
  --terrain competition --cases nominal --heading-mode heading \
  --seeds 0 1 2 --num-envs 8 --duration 15 --goal-m 8 --trace
```

预先设定进入 front 的最低条件：候选短程单回合达标；8m 矩阵至少 22/24 达标、无跌落/基座接触，且达标数不少于相同矩阵下的源模型。此门槛是本实验的推进规则，不代表统计显著性或一般可靠性。若源模型也未达门槛，先分析两组终止原因与轨迹，再调整实验设计。

通过后才从该候选建立独立 front 实验（建议先 100 次更新，再验收，最多逐步扩展到 500 次），并保留 flat 回归。front 评估仍从固定 x=2 出发、前进 22m、60s，不用训练出生点替换评估起点来宣称提升。未通过则停止扩大区间；短程阶段本身仍可能退化，应保留失败模型并排查优化过程。

当前未实现高台阶/梅花桩区域的地形高度感知重置，也未证明整条赛道通过。
