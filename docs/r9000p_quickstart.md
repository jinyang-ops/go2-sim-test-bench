# 联想拯救者 R9000P / RTX 4060 起步指南

已知硬件由项目使用者提供：R9000P、RTX 4060、32GB 系统内存、1TB 存储；系统为 Ubuntu 22.04。

NVIDIA 官方列出的 RTX 4060 Laptop 标准显存为 8GB，最终以本机 nvidia-smi 输出为准。32GB 是系统内存，并不是 GPU 显存。此硬件可作为本项目复现与小规模实验的起点，但当前尚未在这台电脑上测得训练速度、稳定性或收敛表现。

硬件规格依据：[NVIDIA RTX 40 Series Laptop comparison](https://www.nvidia.com/en-au/geforce/laptops/compare/)。

## 第一步：从自己的仓库取得测试工具

```bash
mkdir -p ~/robot_ws
cd ~/robot_ws
git clone https://github.com/jinyang-ops/go2-sim-test-bench.git
cd go2-sim-test-bench
python3 -m robot_test.doctor --output environment.json
conda --version
```

如果目录已存在，在其内部执行 `git pull --ff-only`，不要重复克隆覆盖原有文件。doctor 使用 Python 标准库，尚未装 Isaac Gym 也能执行。缺少 conda 时把“找不到命令”的输出保留下来，先确定安装路线。

这一步收集 GPU/驱动、系统 Python、Conda 与依赖包状态。environment.json 中没有 NVIDIA GPU 信息时，先用 nvidia-smi 检查驱动是否能识别独显。

## 第二步：确认依赖路线

按根目录 README 第 3 节建立独立 go2-test / Python 3.8 环境。当前候选版本为 PyTorch 2.4.0+cu121；是否能在这台电脑上正常工作仍需实际检查。

CUDA 12.1 的 Linux 最低驱动兼容版本表列为 525.60.13；这只是 CUDA 兼容条件，不能单独证明 Isaac Gym、GPU 型号和系统库已全部兼容。无需为了满足一个版本号就把已有更新驱动降级。

依据：[NVIDIA CUDA 12.1 release notes, Table 2](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html)。

安装后应分别验证：Isaac Gym/torch 导入、torch CUDA 可用、物理仿真启动、训练保存模型。最后才判断复现成功。

## 第三步：采用逐步增大的实验规模

|阶段|建议起点|实际验收|
|---|---|---|
|原项目启动检查|64 个并行环境，50 轮|无崩溃、无数值异常、保存并加载模型|
|平地策略训练|256 个并行环境|观察显存占用、训练曲线与回放效果|
|尝试提高吞吐|在 256 稳定后尝试 512|对比吞吐与显存，决定是否保留|
|自动评估|8 个环境/批，评估种子 0、1、2|实际生成每回合数据与失败原因|

这些是实验起点，不是显存上限或保证收敛的配置。显存不足时降低 num_envs，训练期间可在另一个终端运行 `watch -n 1 nvidia-smi` 观察资源使用。训练环境数量和训练更新轮数是两个不同参数。

先依 README 第 4 节完成原项目最小复现，然后使用下列平地基线命令。确认候选依赖和原项目已安装后再执行：

```bash
conda activate go2-test
cd ~/robot_ws/go2-sim-test-bench
python -m robot_test.train --task=go2 --qa-terrain plane --num_envs=256 --headless --max_iterations=1000 --seed=42 --experiment_name=qa_baseline --run_name=r9000p_seed42
```

1000 轮用于开始观察实验，并不等于足够完成任务。实际收敛需根据回放和测试结果判断。后续增加 domain randomization 时保持相同训练预算、评估协议和种子集合，以便公平比较。

## 第四步：积累求职展示证据

按顺序保存：实际安装环境、原项目启动/回放日志、基线与变体模型哈希、五场景批量评估、至少一个真实问题的定位过程、演示视频。

CPU 单元测试可以在线 CI 自动执行。训练、Isaac Gym 集成、策略性能和全赛道结果必须在实际 GPU 环境完成后再标为通过。
