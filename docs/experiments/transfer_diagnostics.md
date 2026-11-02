# 迁移训练离线诊断

原完整范围迁移及后续 flat 分段恢复均未通过短程运动能力保持检查，见[实际结果](../results/competition_transfer_regression.md)。当前停止追加训练与扩大区间；首先检查已有产物。

## 已核对的代码行为

参考上游提交 `7ef08a3f8069750213071adf0032c04cf1be1777`：

- `rsl_rl/modules/actor_critic.py`：训练通过 `act()` 从 Normal(mean,std) 采样，推理通过 `act_inference()` 返回动作均值。推理能走不保证同一策略在探索采样下也稳定。
- `rsl_rl/algorithms/ppo.py`：adaptive 学习率按 KL 调整，代码上限为 1e-2、下限为 1e-5。初始 learning_rate 不等于整个训练过程的学习率。
- `rsl_rl/runners/on_policy_runner.py`：model_0 在第一次更新后保存；中间文件的 iter 字段可能保持为零，到本次 learn() 结束才更新。因此不能用中间文件的 iter 字段将其当成未更新模型。
- 本仓库 `initialize_network_weights` 严格加载完整网络权重，包括 actor、critic、std；不恢复优化器及训练迭代。

这些是代码行为核对，不构成该次退化的根因证明。尤其不能直接把标准差 1.39 定性为故障阈值，或将单次 value loss 解释为模型失效原因。实验端本地代码可能有修改，应同时检查 qa_training.json 中的代码版本和工作树状态。

## 读取现有检查点

在仿真 Python 环境中运行，不创建仿真，不占用 CUDA 计算，不调用 optimizer.step，也不修改模型：

```bash
python -m robot_test.inspect_training \
  --source /absolute/path/to/qa_plane_mass_rand/model_1000.pt \
  --run-dir /absolute/path/to/qa_course_flat_recovery/run \
  --iterations 0 50 100 \
  --evaluation reports/course_flat_100_sanity_v2 \
  --output reports/course_flat_100_diagnostics.json
```

诊断记录包括：

- 本轮初始化记录的源 SHA256 是否匹配实际传入的源文件；这只是记录一致性核对。
- 各检查点完整 SHA256、动作标准差及保存时的优化器学习率。
- actor、critic、std 相对源模型的参数变化，按组汇总 L2 距离及变化元素数；参数名或形状不匹配时明确拒绝部分比较。
- 源模型与目标运行可用的配置记录：seed、命令、sim dt、control decimation、回合长度、归一化、随机化及 PPO 参数。
- 本次评估的协议、结束位置与时长，以及评估文件哈希是否对应本次检查的候选模型。

缺失检查点被明确列出，不伪造对照；全部缺失则失败。JSON 中的 relative_delta_l2 在源参数范数为零时为 null。数值不是性能评分，不能依据参数变化幅度直接判定模型是否能走。保存的优化器学习率仅是保存时刻快照，不能代替完整轨迹。

## 后续对照顺序

1. 先排除初始化哈希、导入路径、配置、评估协议或模型文件不一致。首次更新文件也需要实际评估才能判断何时出现性能退化，参数变化本身不能回答这一点。
2. 若记录一致，再设计不更新权重的动作均值与探索采样对照；保持相同场景、初态与命令，并保留扰动版本标签。
3. 根据这些结果，分别检查探索幅度与优化更新；降低学习率时还应核对是否使用 adaptive 调度，避免把初始学习率当成固定上限。
4. 每次只建立明确可解释的独立对照，短程能力保持通过后再恢复分段扩展。任何修改后的模型都重新使用相同正式评估协议验收。

诊断文件默认位于 Git 忽略的 reports/。共享前检查其中的本地工作树状态与运行配置；源码与文档不包含机器上的绝对检查点路径。CPU 单元测试使用合成夹具，不代表实际模型诊断或 GPU 物理结果。

已有首轮更新检查点的短程评估未达标。下一项使用[固定学习率单轮对照](fixed_lr_probe.md)，保留探索参数与网络初始化，只控制优化器步长策略，并比较保存的更新前后模型。
