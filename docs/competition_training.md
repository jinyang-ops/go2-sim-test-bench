# 比赛地图迁移训练与分段验证

## 参考地图

本页依据上游提交 7ef08a3f8069750213071adf0032c04cf1be1777 的 Go2Robot.create_competition_map、出生点与终止条件。若本地工作树有修改，以实际代码、配置和生成的地形为准。

参考地图宽 12m、长 108m，沿世界 X 排列：

|X 范围 m|路段|
|---|---|
|0–12|平地|
|12–24|负坡度金字塔形坡面|
|24–36|随机起伏|
|36–48|离散障碍|
|48–60|波浪|
|60–72|上台阶|
|72–84|下台阶|
|84–96|梅花桩|
|96–108|保留的平坦尾段|

注意：

- 原训练 competition 出生原点在 x∈[6,48]、y∈[4,8] 采样。出生原点在创建环境时确定，各回合还会随机初速度。不能据此假定后半程训练覆盖充分。
- 原配置每回合 25s，不等于足够从固定起点完成整条路线。
- 原终止条件含基座接触、z<-3 与回合超时，没有自动的“到终点成功”判定。
- 现有 terrain.curriculum 开关不会自动解决 competition 的阶段划分；该环境会关闭对应课程逻辑。
- 上游 SDK 改过下楼梯生成函数的 init_height 参数。训练前保留对应版本并核对工作树。

## 阶段 1：从平地策略初始化地图训练

使用 --qa-init-model 加载完整 ActorCritic 网络权重，严格检查形状；优化器和训练迭代从零开始。初始权重路径、SHA256、源迭代号及配置写入 qa_training.json。

它与 --resume 不同：--resume 恢复已有实验的状态；--qa-init-model 为新任务建立独立实验，二者不能同时使用。若观察维度、动作维度或网络结构改变，权重形状不兼容时应明确失败。

在已有的 go2-test 环境中：

```bash
cd ~/robot_ws/go2-sim-test-bench
git pull --ff-only
init_model=/absolute/path/to/plane/model_1000.pt
mkdir -p reports/competition_transfer
set -o pipefail

python -u -m robot_test.train \
  --task=go2 \
  --qa-terrain competition \
  --qa-init-model "$init_model" \
  --num_envs=128 \
  --headless \
  --max_iterations=2000 \
  --seed=42 \
  --experiment_name=qa_competition_transfer \
  --run_name=seed42 \
  2>&1 | tee reports/competition_transfer/train.log
```

128 环境/2000 轮是初始资源预算，不保证地图收敛。此阶段保持原地图、原奖励和原出生范围，不同时加入新的质量/摩擦扰动。迁移模型可能保留平地能力，也可能在复杂地形失败，需要实际评估。

模型仍位于实际 editable 安装的上游 legged_gym/logs 下；新 model_2000.pt 的编号表示本次迁移训练的更新数，不包含源模型迭代数。

## 阶段 2：固定起点的前段测试

地图评估固定出生原点 x=2、y=6。首先测试前进 22m 到 x≈24，覆盖平地与坡面，目标时限为 60s。示例：

```bash
model_path=/absolute/path/to/competition/model_2000.pt
python -m robot_test.matrix \
  --model "$model_path" \
  --output reports/competition_front_heading \
  --terrain competition \
  --cases nominal \
  --heading-mode heading \
  --num-envs 8 \
  --seeds 0 1 2 \
  --duration 60 \
  --goal-m 22 \
  --trace

python -m robot_test.report reports/competition_front_heading \
  --output reports/competition_front_report
```

若需要测量地图训练增益，应先将初始平地模型放进相同的地图评估协议，并使用独立输出目录。不能用平地历史成绩直接代替地图对照。

CSV 新增 start_x_m、end_x_m、end_y_m、end_z_m 和 terminal_segment。终止位置来自重置前状态；参考路段仅在地图尺寸匹配 12m×108m 的配置时标注，尺寸不同则标为 unverified_layout。路段标签是参考坐标定位，不自动代表该路段已经完成。

## 航向协议必须分开

- yaw_rate：默认历史协议，给定机体 vx 和零 yaw rate，无额外目标航向反馈；协议名 world_x_yaw_rate_v1。
- heading：启用上游原有航向反馈，目标朝向为世界 X 正方向；与原训练配置的 heading_command=True 对齐；协议名 world_x_heading_v2。

评估使用 --qa-heading-mode，矩阵使用 --heading-mode。默认仍是 yaw_rate，已有结果不被改写。报告按模型、地形、场景、目标、时限、速度、航向模式和协议分组，不将两种模式平均在一起。

新增 heading 是独立测试条件，不是对旧模型性能的修复证明。所有模型必须用同一协议重测后才能比较；它也没有新增横向走廊限制。若规格要求直行或车道保持，需要另定义横向偏移和航向阈值，并建立新的验收版本。

## 阶段 3：扩展路段覆盖

前段测试稳定后，按随机起伏/障碍、波浪、上台阶、下台阶、梅花桩逐段扩展。

后续针对后半程需要新增分段出生配置、基于地形高度的安全重置、明确的段末目标和覆盖统计。当前入口尚未实现这些自动分段训练功能；不要只扩大出生 X 范围而仍使用零地形高度，否则可能生成埋入台阶的无效初态。

固定起点的长距离评估可使用目标 94m（x≈96）与相应时限，但只有实际达标才计为该协议的目标完成；不因训练轮数或模型保存而自动标记通关。

## 阶段 4：回归与产物

|测试项|产物与判定|
|---|---|
|模型与环境加载|配置、初始/目标权重哈希、代码版本、导入路径|
|每段能力|固定起点、指令、目标、时限、成功率与前进距离|
|异常定位|重置前终止坐标、参考路段、终止原因与轨迹|
|扰动覆盖|质量、摩擦、噪声、推扰；与 nominal 分开报告|
|能力保持|地图训练后回归平地与已有扰动用例|
|复现性|独立训练种子、固定评估种子、完整 CSV/元数据及变更记录|

新入口已进行 CPU 软件逻辑检查。地图迁移训练和新航向协议的 GPU 结果尚未归档；不填入预期成功率或将计划当成已执行数据。
