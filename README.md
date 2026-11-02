# GO2 四足机器人仿真测试台架

面向机器人测试工程师实习的项目起点：先复现策略训练，再建立可重复评估和失败分析。

[![CPU checks](https://github.com/jinyang-ops/go2-sim-test-bench/actions/workflows/robot-qa.yml/badge.svg)](https://github.com/jinyang-ops/go2-sim-test-bench/actions/workflows/robot-qa.yml)

项目仓库：[jinyang-ops/go2-sim-test-bench](https://github.com/jinyang-ops/go2-sim-test-bench)。硬件起步说明见 [R9000P / RTX 4060 操作指南](docs/r9000p_quickstart.md)。

**验证状态：使用者已在 Ubuntu 22.04 / RTX 4060 Laptop 上完成依赖导入、gymtorch 编译、原项目训练/回放、1000 轮平地训练，以及五场景共 120 回合评估，并提供终端结果。负载场景仍存在明显性能短板。GPU 结果记录见下方链接；完整赛道、ROS 2/Gazebo 接入和实机验证尚未完成。模型权重与原始完整数据尚未上传。**

- [120 回合平地基线报告](docs/results/plane_baseline_matrix.md)
- [50 轮与 1000 轮模型的短程测试](docs/results/short_motion_comparison.md)
- [验证范围与待完成项](VALIDATION.md)

上游：[jindadu00/legged_robot_competition](https://github.com/jindadu00/legged_robot_competition)。本次检查的 master 提交：`7ef08a3f8069750213071adf0032c04cf1be1777`。

这里发布自己的测试台架，保留上游归属；上游代码、Isaac Gym SDK、模型资源放在另一个目录，不随本项目上传。

## 1. 先厘清技术栈

|工具|本阶段作用|
|---|---|
|Ubuntu 22.04|你的宿主系统，需实机兼容性验证|
|Isaac Gym Preview 4|本项目使用的物理仿真与并行训练环境|
|PyTorch / rsl_rl / legged_gym|策略、PPO 训练、机器人环境|
|ROS 2 / Gazebo|当前复现不依赖；后期可另做策略接入及接口测试|
|Isaac Sim|与 Isaac Gym 不同，不能直接替代此仓库的仿真依赖|

作者 README 给出的环境是 Ubuntu 20.04.6、PyTorch 2.4.0+cu121；SDK setup.py 要求 Python >=3.6 且 <3.9，仓库包含 gym_38.so。

NVIDIA 官方目前将 Isaac Gym 列为停止支持的旧版软件，公开前置条件是 Ubuntu 18.04/20.04、Python 3.6–3.8、Pascal 或更新 NVIDIA GPU 且至少 8GB 显存。这个显存要求来自官方，不是本项目的实测下限。22.04 不在官方支持列表内，Conda 只能隔离 Python，不能保证系统/驱动/二进制兼容。

- 官方入口与许可：[Isaac Gym archive](https://developer.nvidia.com/isaac-gym/download)
- PyTorch 对应版本命令：[Previous PyTorch versions](https://pytorch.org/get-started/previous-versions/)

若旧栈不支持你的显卡架构，先调整路线；不要为了这个项目覆盖现有 ROS 2 的系统 Python 或盲目更换驱动。

## 2. 第一次只做环境检查

首次使用先克隆测试仓库，终端执行（已有本地副本时在对应目录执行 `git pull --ff-only`）：

```bash
mkdir -p ~/robot_ws
cd ~/robot_ws
git clone https://github.com/jinyang-ops/go2-sim-test-bench.git
cd ~/robot_ws/go2-sim-test-bench
python3 -m robot_test.doctor --output environment.json
```

这一步不用安装 PyTorch/Isaac Gym。输出包含系统、Python、显卡型号/显存/驱动、Conda 和包路径。把输出或 environment.json 发给协作助手，确定硬件与 CUDA 版本路线后，再做下一节。

如果没有 Conda，先从官方安装 Miniconda/Miniforge；本项目不自动安装或修改你的电脑。没有可用 NVIDIA GPU 时，可在本地完成台架单元测试，训练/仿真需要另外的合适 GPU 环境。

## 3. 候选环境安装（确认显卡/驱动后执行）

建议开一个没有 source ROS 的新终端。保留原来的 ROS 环境。

```bash
mkdir -p ~/robot_ws
cd ~/robot_ws
git clone https://github.com/jindadu00/legged_robot_competition.git
cd legged_robot_competition
git checkout 7ef08a3f8069750213071adf0032c04cf1be1777

conda create -n go2-test python=3.8 -y
conda activate go2-test
python -m pip install pip==24.3.1 setuptools==65.5.1 wheel==0.41.3

cd ~/robot_ws/go2-sim-test-bench
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-sim-py38.txt
```

cu121 是对齐作者环境的候选路线；先确认驱动支持该运行时及 GPU 架构。`nvidia-smi` 的 CUDA Version 是驱动支持的上限，不等于安装的 CUDA Toolkit。若 torch 扩展构建报错，再结合 nvcc/gcc 和错误日志处理，不能只看一个版本号判断成功。

确认你有权使用并已阅读/接受 Isaac Gym 对应许可后，安装上游提供的本地 SDK：

```bash
python -m pip install -e ~/robot_ws/legged_robot_competition/isaacgym/python
python -m pip install -e ~/robot_ws/legged_robot_competition/rsl_rl
python -m pip install -e ~/robot_ws/legged_robot_competition/legged_gym
python -m pip check
python -m robot_test.doctor --deep --output reports/environment_installed.json
```

**上游改过 SDK 的 terrain_utils.py**：下楼梯使用 `stairs_terrain(..., init_height=850)`。直接换成未修改的官方 SDK 可能在 competition 地形报参数错误。本路线安装上游对应版本以保持行为；不要把此 SDK 拷贝进自己的测试仓库。`scipy` 固定为 1.10.1 也保留了上游依赖的 `interp2d`。

`numpy==1.23.5` 用于兼容旧代码里的 NumPy 类型别名。若后续升级 NumPy，应定位具体调用修正并回归验证，避免全仓库替换 `float32` 等合法类型。

如果 deep import 提示 `libpython3.8.so` 找不到，先确认处于 go2-test 环境，可只在当前终端尝试：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

导入通过不代表物理仿真通过。若有段错误、CUDA 架构或动态库错误，保留完整首个错误再定位。22.04 持续不兼容时，可改用独立 20.04 GPU 环境；是否需要由实际报错决定。

## 4. 原项目最小复现

先在原始比赛地形做小规模训练，只验证整个链路是否能启动、保存、回放，不承诺 50 轮能学会行走：

```bash
conda activate go2-test
cd ~/robot_ws/legged_robot_competition/legged_gym/legged_gym/scripts
python train.py --task=go2 --num_envs=64 --headless --max_iterations=50 --experiment_name=repro_smoke --run_name=first
```

显存不足可先降为 16；这是调试规模，不代表足够的训练规模。训练吞吐和收敛耗时需实测。

查看实际保存的权重，不猜测它的编号：

```bash
find ~/robot_ws/legged_robot_competition/legged_gym/logs/repro_smoke -name 'model_*.pt'
```

在带图形桌面的终端回放：

```bash
python play.py --task=go2 --num_envs=1 --experiment_name=repro_smoke --load_run=你的实际运行目录名 --checkpoint=实际模型编号
```

`--load_run` 是 logs/repro_smoke 下那层目录名；`--checkpoint` 是 model_N.pt 中的 N。此版本源码未实现 README 部分示例中的 `--model_dir`，不要直接使用那个参数。无桌面的 headless 训练不产生可视窗口。

完成标准：训练连续运行、无非有限数/崩溃、确实保存 model_*.pt、play.py 能加载对应模型。运动效果另行评估。

## 5. 让它成为自己的测试工程项目

第一版先做平地，降低策略训练难度，再进比赛地形。下面的测试台架不会修改上游工作树。

### 5.1 两组有意义的训练

```bash
cd ~/robot_ws/go2-sim-test-bench
conda activate go2-test

python -m robot_test.train --task=go2 --qa-terrain plane --num_envs=64 --headless --max_iterations=50 --seed=42 --experiment_name=qa_plane --run_name=smoke
```

确认 smoke 通过后再提高到合适的并行数和训练轮数。下面 512 环境/1000 轮只是实验起点，按显存和效果调整；不是保证收敛的配置：

```bash
python -m robot_test.train --task=go2 --qa-terrain plane --num_envs=512 --headless --max_iterations=1000 --seed=42 --experiment_name=qa_baseline --run_name=seed42
python -m robot_test.train --task=go2 --qa-terrain plane --qa-domain-rand --num_envs=512 --headless --max_iterations=1000 --seed=42 --experiment_name=qa_robust --run_name=seed42
```

改造项：在创建环境前同步训练与环境 seed；保存完整配置和代码版本；robust 组启用摩擦随机化、质量随机化、观测噪声及推扰。比较时保持训练预算相同。固定 seed 不能保证 GPU 位级一致，多组训练种子才更能支持改进结论。

### 5.2 单个实际模型评估

先把 **你自己的** 模型文件实际路径赋给变量：

```bash
model_path=/你的实际路径/model_实际编号.pt
python -m robot_test.evaluate --task=go2 --headless --num_envs=8 --seed=0 --qa-model "$model_path" --qa-output reports/baseline_single --qa-terrain plane --qa-case nominal --qa-duration 20 --qa-goal-m 10
```

产物：每回合 episodes.csv、汇总 summary.json、元数据 metadata.json。它记录权重 SHA256、环境/台架 commit 与未提交改动状态、seed、配置、实际机器人形状摩擦值和基座质量。

### 5.3 批量测试与报告

```bash
python -m robot_test.matrix --model "$model_path" --output reports/baseline --terrain plane --num-envs 8 --seeds 0 1 2
```

将 model_path 换成 robust 模型，再执行：

```bash
python -m robot_test.matrix --model "$model_path" --output reports/robust --terrain plane --num-envs 8 --seeds 0 1 2
python -m robot_test.report reports/baseline reports/robust --output reports/comparison
```

每组是 5 场景 × 3 评估种子 × 8 个第一回合 = **计划 120 回合**。这是计划测试量，只有实际生成的结果才算完成。每个场景使用独立进程，错误会保存到 matrix_status.json，缺失结果不会被补成通过。

|场景|实际配置变化|关注点|
|---|---|---|
|nominal|机器人形状摩擦系数固定 1.0，无额外质量/噪声/推扰|基线|
|robot_friction|机器人形状摩擦系数 0.3|接触条件变化；不是测得的有效地面摩擦|
|added_mass|基座增加 2kg|负载敏感度|
|noisy_observation|启用原环境观测噪声|观测鲁棒性|
|push|每 5s 设置随机 XY 基座速度，分量范围 ±0.5m/s|速度扰动响应；不是施加已知力的牛顿值测试|

先失败的回合可能在首次推扰前就结束；不要把它解释为已受到推扰后失败。基线与变体应使用相同评估协议、种子集和权重选择规则。改动训练参数不一定带来提升，未提升也可以做有效的失败分析。

### 5.4 指标定义与正确性

- 成功：在时间上限内前进达到 qa-goal-m，且这一帧没有基座碰撞、跌落或出界。默认是 **20s 内前进 10m**，不是全赛道通过率。
- 失败优先：同一帧同时达到目标和基座碰撞，记为失败。
- 摔倒判断：沿用上游基座终止接触力阈值 >1 和 z<-3 等终止条件；它不是精确的人类姿态标注。
- 前进距离：world x 相对本回合起点的差；记录最大前进和结束位置，不把重置传送当成运动。
- 速度误差：机体坐标系 XY 速度向量误差的 RMS，单位 m/s。
- 关节力矩：策略步采样的各关节绝对力矩均值，单位 N·m；不是能耗指标。
- 采样：SnapshotMixin 在 check_termination 之后、reset_idx 之前克隆状态，防止终止帧被出生位置覆盖。
- 一环境只统计第一回合；已完成环境后续自动重置产生的数据不再计入。到测试时限未达目标记 time_limit，数值异常标为无效运行。
- CLI 可选 `--qa-min-success-rate 0.8` 设回归门槛；80% 是示例，应先根据基线和需求确定。默认只记录结果，不擅自宣称达标。

Competition 评估时固定从 x=2、y=6 起步，而上游默认从 x∈[6,48] 随机出生。默认 10m 只到第一段地形边界。要测试更长路线，可以设 `--qa-terrain competition --qa-goal-m 94 --qa-duration 120`，但这是测试目标，不保证现有策略能完成。完整路线需要实际训练与逐段验证，不能用平地成绩替代。

## 6. 测试用例和缺陷记录

|编号|用例|验收证据|
|---|---|---|
|ENV-01|正确 Python/驱动/包导入|environment_installed.json + 仿真启动日志|
|SIM-01|64 环境原项目训练 smoke|日志、模型路径与实际运行配置|
|SIM-02|模型回放|模型 SHA256、20–60s 本地录屏|
|QA-01|终止帧与重置分离|CPU 单元测试 + 实机终止回合核对|
|QA-02|seed 在创建仿真前同步|qa_training.json、metadata.json|
|QA-03|五场景批量评估|episodes.csv、summary.json、矩阵状态|
|QA-04|观察/动作形状及有限数检查|异常日志或成功评估记录|
|REG-01|baseline 与 robust 等预算对比|真实对比报告、逐种子失败原因|

每个缺陷写：触发条件、实际表现、期望表现、最小复现命令、日志、原因、修复、回归结果。不要把预期结果填成已观察结果。

## 7. CPU CI 与本地验证

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q robot_test tests
```

GitHub Actions 仅运行上述 CPU 检查，不冒充 GPU 仿真测试。指标单元测试和报告测试使用人工构造数据，不能作为策略性能数据。

## 8. 上传到你自己的 GitHub

建议新建独立公开仓库 `go2-sim-test-bench`，本目录作为根目录。不要在上游 SDK 目录里执行 git add .。先保留已有上游版权和链接，展示本仓库的测试代码贡献。

方式 A：在 GitHub 网页创建同名空仓库（先不要勾选 README/LICENSE），然后在本目录执行：

```bash
cd ~/robot_ws/go2-sim-test-bench
git init -b main
git add README.md VALIDATION.md requirements-sim-py38.txt .gitignore .github robot_test tests docs
git commit -m "Add GO2 simulation test bench and CPU checks"
git remote add origin https://github.com/你的用户名/go2-sim-test-bench.git
git push -u origin main
```

方式 B：如果已经安装并在自己电脑登录 GitHub CLI，在完成 git init/add/commit 后运行：

```bash
gh repo create go2-sim-test-bench --public --source=. --remote=origin --push
```

不要把密码或 token 发给协作助手。报告默认被忽略；实际实验完成后，挑选不含本地绝对路径/个人信息的报告摘要存入 docs/results，按正常 git add 提交。保留原始数据及模型哈希以便复查，较大模型用合适的发行附件单独管理。

本项目的远程仓库已确定为 `jinyang-ops/go2-sim-test-bench`。直接克隆本仓库使用即可；上面新建仓库命令只适用于尚无远程仓库的独立副本。后续提交使用已配置的 origin，不要重复添加。

## 9. 简历写法（实际完成后再使用）

项目名：**基于 Isaac Gym 的 GO2 四足机器人仿真与自动化测试**。

“基于开源 legged_gym/rsl_rl 复现 GO2 策略训练与回放，开发 Python 批量评估台架，覆盖基线、接触摩擦变化、负载、观测噪声与推扰场景；通过重置前状态采集计算前进距离、失败原因与速度误差，记录随机种子、配置和权重哈希，生成 CSV/JSON/Markdown 报告，并配置 CPU 单元测试 CI。”

只有完成实际实验后，才补充“执行 X 回合测试、发现并定位 Y 个缺陷、在相同协议下某指标由 A 变为 B”等量化结果。项目尚未做 ROS 2/Gazebo 或实机部署时，不写已完成这些内容。

## 10. 推荐里程碑（完成标准优先于天数）

1. 环境诊断与兼容性 smoke：可以导入、训练、保存、回放。
2. 平地基线策略：先有可用动作，再讨论复杂地形。
3. 自动评估与失败归因：核对终止帧采样，实际执行测试矩阵。
4. 等预算变体实验：比较随机化前后结果，记录失败与无提升情况。
5. 比赛地形逐段测试：更长路线和固定起点，之后再考虑 ROS 2/Gazebo 接入。
6. GitHub 整理：清晰 README、实际结果、复现命令、缺陷案例、演示视频与简历条目。

对大三实习求职，能解释测试设计、数据边界和定位过程，比只展示一段机器人跑动视频更能体现测试能力。
