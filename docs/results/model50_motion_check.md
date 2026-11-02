# model_50 短程前进评估记录

日期：2026-11-02。

## 数据来源与状态

以下数值来自使用者在本机运行 robot_test.evaluate 后提供的终端输出和 summary.json 内容。不是 CPU 单元测试夹具，也不是助手云端运行的数据。尚未取得原始 episodes.csv、metadata.json 或模型文件。

本次只完成一个回合，不能据此估计整体成功率或全赛道通过率。

## 环境与模型

- 使用者系统：Ubuntu 22.04.5 LTS。
- GPU：RTX 4060 Laptop，终端已报告 GPU PhysX、cuda:0、GPU Pipeline enabled。
- Python 3.8；PyTorch 2.4.0+cu121。
- 模型相对上游项目路径：legged_gym/logs/repro_smoke/Nov02_12-38-16_first/model_50.pt。
- 上游创建模型的启动检查参数：64 个环境、50 轮。
- 模型 SHA256、实际上游 commit、完整配置：待从本机原始记录补齐，不推测填写。

## 评估协议

- task=go2，headless，num_envs=1，seed=0。
- qa-terrain=competition，默认 qa-case=nominal。
- qa-speed=1.0 m/s，qa-duration=5 s，qa-goal-m=1 m。
- 此评估入口固定从赛道起点附近出生（x=2、y=6），统计第一个回合；与原版 play 的随机出生条件不同。

## 使用者报告的实际输出

```json
{
  "trials": 1,
  "success_rate": 0.0,
  "mean_max_progress_m": 0.03876662254333496,
  "mean_velocity_vector_rmse_mps": 0.9952516179797145,
  "mean_abs_joint_torque_nm": 3.831489962385905,
  "outcomes": {
    "time_limit": 1
  }
}
```

## 解释与边界

- 最大前进约 3.88 cm；这个回合在时限内未达到 1m 目标。
- 评估程序完成了步进与输出，模型加载和 GPU 仿真入口已运行；这支持当前策略前进能力不足的判断，不能把单张静止画面直接视为整个仿真卡死。
- 3.83 N·m 是按策略步采样的平均绝对关节力矩，不是策略动作幅度、机械功率或能耗。
- time_limit 是本次评估结果；不代表机器人完成了任务。
- 日志中原版 play 的 Matplotlib Agg 提示涉及图表显示；无窗口评估仍正常结束。本记录不把这些 warning 判为导致本次前进失败的根因。
- 单回合结果不足以排除所有控制/动力学问题，也不足以证明增加训练轮数必然解决问题。

## 后续实验

1. 从头训练平地基线，候选起点为 256 个环境、1000 轮、训练 seed=42；按实际资源与训练曲线调整。
2. 用新模型重复同一个 5s / 1m、competition、seed=0 测试，先确认基本前进能力。
3. 再扩大种子数和测试场景；平地与复杂赛道结果分别报告。
4. 从本机保存并核对原始 CSV、元数据、依赖版本、模型哈希，再补充简历量化结果。
