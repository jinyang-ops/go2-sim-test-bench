# 参考环境与安装检查

## 已记录的运行环境

|组件|参考配置|
|---|---|
|系统|Ubuntu 22.04.5 LTS，x86_64|
|GPU|NVIDIA RTX 4060 Laptop，约 8GB 显存|
|驱动|595.91.07（实验记录）|
|Python|独立 Conda 环境中的 Python 3.8|
|训练框架|PyTorch 2.4.0+cu121|
|仿真|Isaac Gym Preview 4，上游配套修改版本|
|内存|诊断输出约 15GiB 可见内存|

该配置已产生平地 GPU 训练与评估汇总；不表示其他驱动、硬件或系统组合自动兼容。Isaac Gym 官方列出的系统为 Ubuntu 18.04/20.04，Python 3.6–3.8。

安装命令见根目录 README。既有上游工作树可保留原路径，editable 安装后通过 doctor 确认实际导入目录。

## 诊断

```bash
conda activate go2-test
cd ~/robot_ws/go2-sim-test-bench
python -m robot_test.doctor --output reports/environment_before_import.json
python -m pip check
nvidia-smi
```

Isaac Gym 必须在 torch 前导入。gymtorch 首次导入会编译 C++ 扩展；MAX_JOBS=2 可限制并行编译数量。

## 动态库路径

```bash
ls -l "$CONDA_PREFIX/lib/libpython3.8.so.1.0"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2
python -m robot_test.doctor --deep --output reports/environment_after_import.json
```

上述导出作用于当前终端。导入、GPU 张量计算、物理仿真、模型保存和回放需要分别检查；导入成功本身不证明全部环节通过。

## 资源规模

|任务|初始规模|
|---|---|
|启动检查|64 环境、50 轮|
|平地训练|256 环境、1000 轮|
|比赛地图迁移|先小规模验证，再按显存和吞吐调整|
|回归矩阵|8 环境/批、种子 0/1/2|

这些是实验起点。实际显存占用、速度和收敛表现通过日志测量。保留系统环境与独立训练环境的边界，依赖变更应记录并回归验证。

## 参考资料

- [NVIDIA Isaac Gym archive](https://developer.nvidia.com/isaac-gym/download)
- [PyTorch historical versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 12.1 release notes](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html)
