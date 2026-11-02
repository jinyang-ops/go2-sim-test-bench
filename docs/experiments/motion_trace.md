# 低摩擦超时：世界坐标与机体坐标诊断

## 已观察结果

实验汇总中的汇总如下：

|模型/场景|达标|超时|跌落或基座接触判定|
|---|---:|---:|---:|
|原模型 / added_mass|4|5|15|
|原模型 / robot_friction|22|2|0|
|仅质量随机化 / added_mass|13|11|0|
|仅质量随机化 / robot_friction|13|11|0|

因此本次质量随机化模型的负载跌落/接触判定已从 15 次降至 0；未达标负载回合及低摩擦回合均为超时。不能把低摩擦退化描述为已观察到摔倒增多。

## 待验证的解释

评估的速度误差计算使用 env.base_lin_vel（机体坐标 XY），前进目标使用 world X 相对位移。若机器人朝向偏转，机体向前速度不等于世界 X 速度。当前评估 heading_command=False，给定 body vx=1、vy=0、yaw rate=0；没有额外的世界朝向反馈指令。

这是待核查的任务/坐标解释，不是已确认的失败根因。也可能存在速度跟踪不足或其他因素，需要新采样证据。

## 新增记录

- 单次评估：--qa-trace；矩阵：--trace。默认不记录。
- motion_trace.csv 只保存第一个计分回合，在自动重置前采样，达标/失败后不继续计入。
- 记录：时间、世界 XYZ、相对前进/横向位移、世界/机体 XY 速度、指令、yaw、平均绝对力矩。
- yaw 从 Isaac Gym root state 的 xyzw 四元数计算，相对世界 X 正方向。
- 原有命令、物理环境、时限计数、奖励、目标和失败判定保持。新输出放在独立目录；GPU 重跑不保证位级一致，不能把旧汇总与新轨迹强行合成同一回合。
- 收集记录会增加 CPU 内存和数据处理开销，不用于比较训练或仿真吞吐。

## 执行：只复测新模型的低摩擦场景

在已激活、已配置动态库路径的 go2-test 终端执行：

```bash
cd ~/robot_ws/go2-sim-test-bench
git pull --ff-only
model_path="$HOME/legged_robot_competition/legged_gym/logs/qa_plane_mass_rand/Nov02_13-25-02_seed42/model_1000.pt"

python -m robot_test.matrix \
  --model "$model_path" \
  --output reports/mass_friction_trace \
  --terrain plane \
  --cases robot_friction \
  --num-envs 8 \
  --seeds 0 1 2 \
  --duration 20 \
  --goal-m 10 \
  --trace

python -m robot_test.trace reports/mass_friction_trace \
  --output reports/mass_friction_trace_summary
cat reports/mass_friction_trace_summary/trace_report.md
```

共计划额外记录 24 个首回合。新结果尚未执行，不能把“可能偏航”写成已证明结论。

## 如何解释

- 机体 vx 接近指令、世界 vx 明显较低，且偏航/横向位移较大：朝向偏离可能影响世界 X 目标。
- 两种前进速度都较低、偏航较小：优先继续检查速度跟踪。
- 单凭这些汇总不能认定脚滑；脚滑需额外结合足端速度与接触状态。
- 若需要调整 heading_command 或目标定义，应另建评估协议、明确坐标要求并重测所有待比较模型，不用改判定来抬高当前成绩。
