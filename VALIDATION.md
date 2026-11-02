# 验证记录

日期：2026-11-02。

## 已执行

- 本次云端执行环境：Ubuntu 24.04.3、Python 3.12.14、未安装 PyTorch/Isaac Gym，无可用 NVIDIA GPU。这个环境不是用户的 Ubuntu 22.04 电脑。
- 初始 `python3 -m unittest discover -s tests -v`：14 项通过。覆盖失败优先级、终止后数据冻结、距离/速度误差/力矩统计、时限、不合法数值、场景配置，以及报告协议隔离与空结果拒绝。
- `python3 -m compileall -q robot_test tests`：通过。
- 全部 11 个 Python 文件经过 `ast.parse(..., feature_version=(3, 8))` 语法检查：通过。仅证明使用了 Python 3.8 可识别语法，不证明在 Python 3.8 下的所有运行时行为已验证。
- `python3 -m robot_test.doctor --output /tmp/go2-doctor-smoke.json`：成功生成诊断 JSON；未执行 deep import。
- 新增训练配置隔离/互斥测试后，本地 18 项 CPU 测试和远程 Python 3.8 CI 已通过，见提交 944f9b6 的 Actions 记录。
- 已阅读上游 GO2 配置、任务注册、环境、训练/回放入口、CLI 参数和策略加载逻辑；核对了重置前采集位置以及 SDK 自定义 init_height 参数。

## 使用者本机执行并提供的结果

- Ubuntu 22.04.5、RTX 4060 Laptop、Python 3.8、PyTorch 2.4.0+cu121。
- torch CUDA 计算、Isaac Gym 导入和 gymtorch C++ 扩展编译已通过。
- 原项目 64 环境 / 50 轮启动训练保存了 model_50.pt，play 已加载该模型并打开仿真窗口。
- 新增训练入口完成 256 环境 / 1000 轮平地训练，日志报告累计 12,288,000 环境步、Total time 493.31s，并保存 model_1000.pt。
- model_50 的 5s / 1m 单回合评估未达到目标；model_1000 的对应单回合达标。
- 五场景、3 个评估种子、每批 8 个首回合，共 120 回合平地评估已产生报告：nominal 70.8%、added_mass 16.7%、noisy_observation 83.3%、push 83.3%、robot_friction 91.7%。
- 仅质量随机化组以相同训练预算完成，日志 Total time 496.01s；另完成五场景 120 回合。两组矩阵累计 240 回合。变体：added_mass 54.2%、nominal 95.8%、noisy_observation 100.0%、push 95.8%、robot_friction 54.2%；低摩擦存在退化。
- 已从使用者汇总核对原基线：nominal 7 次超时；added_mass 15 次跌落/基座接触判定、5 次超时；实际基座质量差为 2kg。
- 上述 GPU 数据来自使用者提供的终端和报告，不是助手云端独立执行。已取得汇总，尚未取得完整 CSV/元数据或权重文件用于独立核对。
- 远程仓库和 CPU CI 已建立；CI 状态以 Actions 页面为准，CPU 测试不替代 GPU 物理测试。

## 待完成

- 核对新模型的负载与低摩擦逐种子失败原因、实际属性和回合状态。
- 实际终止帧与 CSV 指标的人工抽查，以及完整原始记录/模型 SHA256 的核对。
- 完成失败归因和必要的额外训练种子验证；当前已观察到负载改善、低摩擦退化，不能泛化为统一改进。
- 更长比赛路线、ROS 2/Gazebo 接入、真实机器人部署和整机测试。

结果记录：[240 回合质量随机化回归](docs/results/mass_randomization_comparison.md)、[平地 120 回合基线](docs/results/plane_baseline_matrix.md)、[短程对比](docs/results/short_motion_comparison.md)。

tests 中数据是人工构造的单元测试夹具；docs/results 中本机 GPU 汇总记录与其分开。单次或少量种子的结果不证明所有任务已达标。
