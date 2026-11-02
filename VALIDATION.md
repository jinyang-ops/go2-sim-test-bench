# 验证记录

日期：2026-11-02。

## 已执行

- 本次云端执行环境：Ubuntu 24.04.3、Python 3.12.14、未安装 PyTorch/Isaac Gym，无可用 NVIDIA GPU。这个环境不是用户的 Ubuntu 22.04 电脑。
- `python3 -m unittest discover -s tests -v`：14 项通过。覆盖失败优先级、终止后数据冻结、距离/速度误差/力矩统计、时限、不合法数值、场景配置，以及报告协议隔离与空结果拒绝。
- `python3 -m compileall -q robot_test tests`：通过。
- 全部 11 个 Python 文件经过 `ast.parse(..., feature_version=(3, 8))` 语法检查：通过。仅证明使用了 Python 3.8 可识别语法，不证明在 Python 3.8 下的所有运行时行为已验证。
- `python3 -m robot_test.doctor --output /tmp/go2-doctor-smoke.json`：成功生成诊断 JSON；未执行 deep import。
- 已阅读上游 GO2 配置、任务注册、环境、训练/回放入口、CLI 参数和策略加载逻辑；核对了重置前采集位置以及 SDK 自定义 init_height 参数。

## 待用户 GPU 环境实际验证

- 候选依赖能否安装并在 Ubuntu 22.04 导入。
- Isaac Gym 原生物理仿真是否启动，GPU 扩展是否能编译。
- 原项目训练、模型保存及 play 回放。
- 新增 train/evaluate 入口与真实 Isaac Gym/torch 的运行时集成。
- 指标与实际终止帧人工核对。
- 五场景 × 多种子实际评估、变体是否改善及更长比赛路线表现。
- 本机训练相关结果尚未产生。远程仓库已确定为 `jinyang-ops/go2-sim-test-bench`；远程 CPU CI 状态以仓库 Actions 页面为准。

所有性能数字必须来自后续真实执行；计划中的 120 回合不能写为已完成。tests 中数据是人工构造的单元测试夹具。
