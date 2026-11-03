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
