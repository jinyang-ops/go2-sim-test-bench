# 比赛地图迁移后的短程运动退化

日期：2026-11-02。数据来源为实验端提供的训练日志、两个 summary.json 及 qa_training.json 初始化字段；完整原始文件和迁移后权重尚未归档，未在 CI 独立复现。

初始策略为平地仅质量随机化训练的 model_1000.pt。新实验 qa_competition_transfer 使用 competition 原随机出生范围、128 环境、seed=42、2000 次更新，共 12,288,000 环境步；优化器和训练迭代重新开始。

## 固定协议对照

三组短程测试均使用 competition 固定出生原点 (2,6,0)、nominal、heading 航向模式、seed=0、单环境、指令速度 1m/s、时限 5s、世界 X 前进目标 1m。该测试位于地图平坦起始段，不代表坡面或全赛道完成。

|模型|报告目录|达标回合/回合数|最大前进 m|机体 XY 速度向量 RMS 误差 m/s|平均绝对关节力矩 Nm|结束原因|
|---|---|---:|---:|---:|---:|---|
|初始质量随机化 model_1000|course_sanity_initial_v2|1/1|1.010695|0.262079|13.747390|goal_reached|
|地图迁移 model_2000|course_sanity_2000_v2|0/1|0.044520|1.016359|16.783604|time_limit|
|flat 分段恢复 model_100|course_flat_100_sanity_v2|0/1|0.009599|1.224373|20.176036|fall_or_base_contact|

只报告本次单回合结果，不把 1/1 解释为一般成功率 100%。力矩均值不等于能耗。

## 初始化核对

- 源实验：qa_plane_mass_rand，运行 Nov02_13-25-02_seed42，model_1000.pt。
- 源 SHA256：`311924e008f8456f1a1ed668af15f6205e6699154a63c6baf1877a38c1b13887`。
- source_iteration=1000，optimizer_restored=false，training_iteration_restored=false。
- 目标实验：qa_competition_transfer，运行 Nov02_19-08-58_seed42，model_2000.pt。

日志显示正确记录了源模型初始化；本次对照不支持“误用未训练模型”这一解释。网络初始化包含 actor、critic 和动作标准差参数，严格匹配形状。

最后一轮训练日志记录 Mean reward=0.05、rew_tracking_lin_vel=0.0055、value function loss=217.0389、Mean action noise std=1.66。这些训练指标不能直接代替运动评估，也不能单独证明退化根因。

## 处置与待验证解释

当前证据支持地图迁移后在该短程条件下运动能力退化。跨地形和出生状态跨度较大是待验证因素；优化过程、critic 迁移及探索噪声也需通过独立对照排查，不能仅凭这组结果断言。

原处理方案是保留 model_2000 作为失败样本，从上述原始 model_1000 重新初始化，只训练起始平坦区间 100 轮，检查基本运动能力后再扩展到前段坡面。配置和准入条件见[短程分段训练](../experiments/course_stages.md)。该方案的后续结果如下。

## flat 分段恢复结果

后续实验 qa_course_flat_recovery / Nov02_20-03-02_seed42 已完成 128 环境、100 次更新，共 614,400 环境步，日志用时 52.95s。最后一轮 action noise std=1.39、rew_tracking_lin_vel=0.0121、stage_goal_fraction=0。后者是重置批次中的训练统计，不是正式评估成功率。

相同短程检查下，model_100 前进约 0.0096m 后触发 fall_or_base_contact。仅凭 summary 不能确定触地部位、首次触地时间或参数更新根因，也不能断言由探索标准差单独导致。完整权重哈希与原始轨迹尚待归档；本次初始化记录也需单独核对，不能借用上一次迁移的初始化核对结果。

该分段恢复方案未通过验收，暂停追加训练及 front 扩展，保留权重与输出作为失败样本。下一步读取已有 model_0/model_50/model_100 的参数变化、动作标准差、保存的优化器学习率和本次训练元数据，见[诊断流程](../experiments/transfer_diagnostics.md)。这不会训练新模型，也不会修改历史评估协议。
