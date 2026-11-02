# GO2 Simulation Test Bench

[![CPU checks](https://github.com/jinyang-ops/go2-sim-test-bench/actions/workflows/robot-qa.yml/badge.svg)](https://github.com/jinyang-ops/go2-sim-test-bench/actions/workflows/robot-qa.yml)

基于 Isaac Gym、legged_gym 和 rsl_rl 的 GO2 仿真训练与回归测试工具。支持平地/比赛地图训练、模型回放、扰动测试、终止前状态采样和实验溯源。

上游项目：[jindadu00/legged_robot_competition](https://github.com/jindadu00/legged_robot_competition)，参考提交：`7ef08a3f8069750213071adf0032c04cf1be1777`。上游源码、模型资源及 SDK 安装在独立目录，不随本仓库分发；使用时保留各组件许可证。

## 功能与验证范围

- 环境诊断：系统、GPU、Python 和依赖检查。
- 训练：平地或 competition 地形；固定 seed；质量随机化与多扰动配置；完整参数记录。
- 评估：nominal、robot_friction、added_mass、noisy_observation、push 五个场景，独立进程、多种子批量执行。
- 数据：重置前采样；每回合 CSV；配置与模型 SHA256；JSON/Markdown 报告；可选运动轨迹。
- 回放：近距离跟随相机；独立的演示与计分入口。
- CPU 检查：指标、配置隔离、命令互斥、轨迹与报告测试。GPU 训练不在普通托管 CI 中运行。

已归档两组训练的 240 回合平地评估汇总，以及额外轨迹诊断。质量随机化组的负载成功率从 16.7% 升至 54.2%，低摩擦成功率从 91.7% 降至 54.2%。结果只代表当前实验协议与种子；完整复杂赛道通过、ROS 2/Gazebo 接入和实机验证尚未完成。权重、完整原始 CSV 和元数据尚未归档，现有汇总不能代替独立复核。

- [验证记录](VALIDATION.md)
- [环境配置](docs/environment.md)
- [比赛地图迁移与分段验证](docs/competition_training.md)
- [平地基线](docs/results/plane_baseline_matrix.md)
- [质量随机化设置](docs/experiments/mass_randomization.md)
- [两组模型回归对比](docs/results/mass_randomization_comparison.md)
- [运动轨迹诊断](docs/experiments/motion_trace.md)
- [缺陷记录模板](docs/defect_template.md)

## 环境

参考 GPU 环境为 Ubuntu 22.04.5、Python 3.8、RTX 4060 Laptop、PyTorch 2.4.0+cu121。Isaac Gym Preview 4 官方前置条件列出 Ubuntu 18.04/20.04 与 Python 3.6–3.8；22.04 的兼容性需要通过实际仿真验证，不能仅凭包安装成功判断。

- [Isaac Gym archive 与许可](https://developer.nvidia.com/isaac-gym/download)
- [PyTorch 历史版本安装命令](https://pytorch.org/get-started/previous-versions/)

此仓库使用 Isaac Gym，不是 Isaac Sim；当前入口不依赖 ROS 2 或 Gazebo。保留系统 Python，使用独立 Conda 环境。

## 安装

```bash
mkdir -p ~/robot_ws
cd ~/robot_ws
git clone https://github.com/jinyang-ops/go2-sim-test-bench.git
git clone https://github.com/jindadu00/legged_robot_competition.git
git -C legged_robot_competition checkout 7ef08a3f8069750213071adf0032c04cf1be1777

conda create -n go2-test python=3.8 -y
conda activate go2-test
python -m pip install pip==24.3.1 setuptools==65.5.1 wheel==0.41.3
cd ~/robot_ws/go2-sim-test-bench
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-sim-py38.txt
```

已有上游工作树时，先核对 commit 与未提交改动，再选择版本，不覆盖已有实验代码。确认 SDK 使用许可后安装对应版本：

```bash
upstream_dir="$HOME/robot_ws/legged_robot_competition"
python -m pip install -e "$upstream_dir/isaacgym/python"
python -m pip install -e "$upstream_dir/rsl_rl"
python -m pip install -e "$upstream_dir/legged_gym"
python -m pip check
python -m robot_test.doctor --deep --output reports/environment.json
```

上游已有工作树也可位于其他目录，editable 安装决定实际导入路径。上游 SDK 的楼梯函数使用自定义 `init_height` 参数；直接换用未修改的 SDK 可能在 competition 地形生成时报错。SciPy 1.10.1 保留旧地形生成依赖的 interp2d；NumPy 1.23.5 用于兼容旧类型别名；Ninja 1.11.1.3 避免 1.11.1.1 的 wheel 元数据错误。

若动态链接器找不到 libpython3.8.so.1.0，先确认文件存在，再在当前终端设置：

```bash
ls -l "$CONDA_PREFIX/lib/libpython3.8.so.1.0"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2
```

## 训练与回放

初始平地训练示例：

```bash
python -m robot_test.train --task=go2 --qa-terrain plane --num_envs=256 --headless --max_iterations=1000 --seed=42 --experiment_name=qa_plane_baseline --run_name=seed42
```

仅质量随机化使用 `--qa-mass-rand`；全部训练扰动使用 `--qa-domain-rand`，二者互斥。预算只是实验起点，不保证收敛。模型默认保存于实际 editable 安装的上游 `legged_gym/logs/<experiment>/<run>/` 中，训练记录为 qa_training.json。

迁移到比赛地图可使用 `--qa-terrain competition --qa-init-model /absolute/path/to/model.pt`。该入口只初始化网络权重，优化器与训练迭代从零开始；不能同时传 `--resume`。新地图入口的 GPU 训练效果待实际运行验证。

```bash
model_path=/absolute/path/to/model_1000.pt
python -m robot_test.play --model "$model_path" --seconds 60 --case nominal --heading-mode heading
```

演示使用航向保持时，与历史 yaw-rate=0 的评估协议不同，不将演示直接当作原协议成绩。

比赛地图回放可附加 `--terrain competition`，相机仍跟随机器人。回放不是计分评估。

## 评估与报告

```bash
python -m robot_test.matrix --model "$model_path" --output reports/baseline_matrix --terrain plane --num-envs 8 --seeds 0 1 2 --duration 20 --goal-m 10
python -m robot_test.report reports/baseline_matrix --output reports/baseline_report
```

默认 5 场景 × 3 种子 × 8 个首回合，共计划 120 回合。输出目录必须是新目录；只有实际产生且有效的数据才计入报告。

新评估可显式使用 `--qa-heading-mode heading`（矩阵使用 `--heading-mode heading`）启用目标朝向反馈。默认 `yaw_rate` 保持旧协议；报告会将两种模式分开，不能把新航向模式下的成绩当成旧协议提升。比赛地图 CSV 另外记录重置前终止位置与参考路段。

- 成功：在时限内首次达到指定 world-X 前进距离，且该帧未被判定失败。
- 失败：基座接触/跌落、赛道出界或到时未达目标；同帧失败优先于达标。
- 速度误差：机体坐标 XY 速度向量 RMS；与世界 X 位移不是同一指标。
- 关节力矩：策略步采样的绝对力矩均值，不是能耗。
- 数据在 reset_idx 前克隆，避免出生点覆盖终止帧；只计第一回合。
- 默认未设成功率门槛时，脚本正常退出不等于任务全部达标。
- 固定评估种子不保证 CUDA 位级一致；各训练种子的结果应分别保存。
- 可选 `--trace` 生成轨迹；`python -m robot_test.trace <matrix_dir> --output <report_dir>` 汇总世界/机体速度、偏航与横向偏移。
- 已观察到名义 20s 的部分离散步记录为 20.02s，边界需结合 dt/valid_steps 核对；当前历史比较使用相同计数实现。

## CPU 检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q robot_test tests
```

tests 中的合成夹具只验证软件逻辑；docs/results 中的 GPU 汇总单独记录。完整实验应保留模型哈希、配置、代码 commit、工作树改动、原始数据与失败案例。报告默认位于被 Git 忽略的 reports/；对外归档前清理本机绝对路径与凭据信息。
