# GO2 仿真复现

- 上游项目：[legged_robot_competition](https://github.com/jindadu00/legged_robot_competition)
- 作者参考项目：[Training-Quadruped-Robots-to-Traverse-Diverse-Complex-Terrains](https://github.com/jindadu00/Training-Quadruped-Robots-to-Traverse-Diverse-Complex-Terrains)

## 从赛道起点训练

`patches/go2-course-start.patch` 针对上游 GO2 的 competition 地图：

- 将环境原点从随机 X=6～48、Y=4～8 米改为 X=2 米、Y=赛道宽度的一半。默认宽度 12 米，因此 Y=6 米。
- 初次创建机器人时取消位置扰动，并加上配置的机体离地高度；默认机体位置为 `(2, 6, 0.42)` 米。
- 回合重置沿用上游对环境原点的使用，因此训练和播放都从同一起点区域开始。
- 将初始相机对准起点。

奖励、网络、地图、命令速度、关节和速度随机化保持上游设置。默认每回合仍为 25 秒；6000 次训练迭代不代表完成整条赛道。

在已安装好 Isaac Gym、PyTorch 和上游依赖的环境中执行：

```bash
conda activate go2-test
cd ~/robot_ws/go2-sim-test-bench
git pull --ff-only && bash scripts/train_from_start.sh
```

上游默认位于 `~/legged_robot_competition`，也可作为脚本第一个参数传入。脚本先应用补丁，然后用 4 个环境检查初次创建位置和两次重置位置。检查通过后，从头训练 6000 轮，使用 256 个并行环境。任何步骤失败都会停止；重复应用同一补丁不会重复修改代码。先关闭已有播放或训练进程，再运行脚本。

模型保存在上游 `legged_gym/logs/course_start_6000/` 下；运行日志、上游提交号和配置代码差异保存在本仓库被 Git 忽略的 `reports/` 下。脚本成功结束时输出本次 `model_6000.pt` 的位置。未加载此前的 50 轮检查点。

本补丁已检查可应用及 Python 语法；GPU 运行检查由脚本在本机执行。训练效果和全赛道通过能力尚未验证。仿真器、训练模型与运行日志不包含在本仓库中。

## 排查坡面停滞和自动重置

`scripts/diagnose_course.py` 读取一个检查点，观察单个环境的第一个回合。使用当前安装的上游 GO2 和起点补丁，沿用上游播放时的命令采样，并关闭观察噪声、摩擦随机化和推扰。脚本不训练、不保存模型，也不修改上游文件。

```bash
python scripts/diagnose_course.py --model /path/to/model_6000.pt --seconds 25 --headless
python scripts/diagnose_course.py --model /path/to/model_6000.pt --seconds 60 --headless
```

两次运行均使用种子 1；每次只在内存中设置回合时限。去掉 `--headless` 可打开窗口。单次运行在首次重置后结束，终端每约 5 秒输出 `DIAG_STEP`，最后输出一行 `DIAG_RESULT`。时间单位为仿真秒。

位置、速度和终止标志在上游自动重置前采样。结果包含 25 秒时的状态（此前终止则为 null）、终止状态和最大世界 X 前进距离。上游重置原因分别为 `time_limit`（超时）、`base_contact`（机体接触力超过阈值）和 `below_height_limit`（世界 Z 小于 -3 米）；同一帧可能同时触发多项。机体坐标速度不等于沿赛道世界 X 的速度，接触终止也不能单独证明控制器故障。

延长播放时间不会更新策略。若 60 秒仍停在同一路段，不能把问题仅归因于 25 秒时限；若提前接触终止，应检查对应位置和姿态。两次单回合只用于排查，不能作为通过率或训练改进的统计证据；GPU 重跑不保证位级一致。本脚本已做 Python 3.8 语法兼容检查和重置前采样的模拟测试，实际 Isaac Gym GPU 运行需在本机验证。

