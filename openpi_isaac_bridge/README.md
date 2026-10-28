# Isaac Lab / OpenPI 相机与观测检查

用于已能运行 `IsaacContrib-Lift-Cube-Franka-IK-Rel` 的 Isaac Lab 3 开发版本。
脚本采集两台相机和 8 维状态；可选请求 `pi05_libero` 模型，但只保存动作，不执行模型动作。
它还不是闭环抓取控制器，尚未校准 LIBERO 与 Isaac 的坐标和动作映射。

用户已在 Spark 验证图片采集、模型查询、XYZ 平移与夹爪开合。新增旋转测试目前只通过本地 CPU 检查，仍需在 Spark 验证。
这部分不需要运行双摆安装脚本，也不修改已有双摆任务。

## 准备

在 Spark 拉取本仓库：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
```

保持 OpenPI 服务运行（在单独的终端和 OpenPI 环境中）：

```bash
cd /home/test/Robot/openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

`--query` 使用 Isaac Lab 环境内的客户端。若之前尚未安装，在 IsaacLab 目录执行：

```bash
cd /home/test/Robot/IsaacLab
uv add --optional openpi --editable /home/test/Robot/openpi/packages/openpi-client
```

若出现依赖冲突，先处理兼容性，不要强行降级已有 Isaac Lab 环境。

## 先保存图片与状态

停止此前的 zero_agent/play 进程，然后执行：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser --extra openpi isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser
```

源码版 Isaac Sim 需要通过 `isaaclab -p` 启动脚本，以激活 `_isaac_sim` 指向的运行环境；
直接使用 `uv run python` 可能提示 Isaac Sim 不在 PYTHONPATH。
本机版本的启动参数不支持 `--headless`，不要添加该参数；这里用 `--viz viser` 选择网页可视化。
脚本使用框架的零/保持策略推进相机预热，并显式保持夹爪张开（本任务 +1 是张开，0/负值是闭合），完成后退出。
默认输出在 `/home/test/Robot/IsaacLab/logs/openpi_camera_check/`：

- `external_camera.png`：外部相机。
- `wrist_camera.png`：手腕相机。
- `observation.json`：状态、坐标约定、手指名称及图像统计。

检查物体与夹爪可见，图像没有全黑、遮挡或方向错误。
脚本会在创建场景前关闭命令目标与末端坐标调试标记，避免彩色箭头/球体进入输入图像。
外部相机默认更靠近桌面；光照默认设为 1000，可通过 `--light-intensity` 调整。
腕部相机向手掌外侧偏移，预热后用物体真实位置进行一次初始视角校准，后续不持续追踪物体。
这属于采集诊断，不能作为部署时的在线相机追踪逻辑。实际闭环前应确认并固定安装外参。
腕部安装位置是初始建议值，需要根据实际图片调整。`--wrist-offset x y z` 使用手部局部坐标，
`--external-eye x y z`、`--external-target x y z` 使用环境坐标；参数会写入 `observation.json`。
不要直接照搬 LIBERO 的图像旋转操作。

8 维状态是机器人根坐标系中的末端位置 3、旋转向量 3、两指关节位置 2。
这只定义了本脚本的输入格式，并不表示已经匹配 LIBERO 的参考坐标与夹持点。

## 再请求一次模型

确认图片正确，且 OpenPI 服务已启动后执行：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser --extra openpi isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --query
```

默认连接 `127.0.0.1:8000`，指令为 `pick up the cube`。
可通过 `--server-host`、`--server-port`、`--prompt` 和 `--output` 覆盖。
成功后保存 `predicted_actions.npy` 并打印 `(动作序列长度, 7)`，不会将这些动作发送给机械臂。

如果 API 不兼容，请保留完整 traceback；不要仅凭状态维度正确就接入执行。
闭环控制前还需验证相机视角、状态坐标、LIBERO 控制器缩放、Isaac IK 坐标与缩放、
夹爪开合方向、控制周期及重置时的动作队列清理。

## 小幅运动与夹爪标定（不执行模型动作）

这是确定性仿真测试，与 `--query` 不能同时使用。每次启动都会重新初始化环境。
先验证 +X 方向，默认每步向相对 IK 目标增加 1 mm，共 10 个控制步，随后短暂保持：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser --extra openpi isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test x
```

将 `x` 换成 `y` 或 `z`，分别检查正方向；换成 `gripper`，执行张开、闭合、再张开。
`delta_root_m` 是末端实际位移（机器人根坐标系，单位米），不是目标值。
相对 IK 在每步按当前位置生成目标，伺服响应会影响累计位移，因此不保证实际移动 10 mm。
单次测试超过 5 cm、状态无效或环境提前重置时会停止；动作按实际 IK scale 换算。
日志与对应 `motion_x.json` / `motion_y.json` / `motion_z.json` / `motion_gripper.json` 保存测量结果。
标定阶段不要求接触方块，夹爪测试也不证明能抓稳。

CPU 上可检查小幅指令及其输入范围保护：

```bash
python3 openpi_isaac_bridge/test_motion_probe.py
```

这些检查不验证 GPU、相机或物理运动。真实运动仍需在 Spark 检查。

## 下一步：小角度旋转验证

XYZ 和夹爪通过后，分别运行 `rx`、`ry`、`rz`，每次重新初始化环境：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test rx
```

把最后的 `rx` 分别换成 `ry`、`rz`。无需启动 OpenPI 服务。
每个控制步向当前 IK 姿态增加正向 0.5° 目标，执行 10 步，再保持 5 步。
按实际 IK scale 换算为弧度指令，夹爪保持张开。
这不是固定的 5° 目标，也不保证累计转动 5°；请以测量值判断响应。
超过 15° 实际旋转、5 cm 实际位移，或环境重置时停止。

输出保存为 `logs/openpi_camera_check/motion_rx.json`、`motion_ry.json`、`motion_rz.json`。
重点检查：

- `delta_rotation_root_deg`：实际相对旋转向量，按机器人根坐标系 X/Y/Z 排列。相应轴应为正且占主导。
- `delta_root_m` 和 `translation_drift_norm_m`：旋转伴随的位置漂移。
- `requested_step_deg`、`ik_scale`、`raw_isaac_action`：目标与实际输入。

相对旋转由 `q_after * inverse(q_before)` 计算，不直接相减两个绝对旋转向量。
四元数数组顺序沿用安装的 Isaac Lab 数学 API；本项目当前 Lab 3 为 xyzw。
通过这一步仅确认 Isaac 相对 IK 的方向与响应，尚未验证 LIBERO 控制器缩放或模型抓取能力。

## 固定目标跟踪验证

平移与旋转方向通过后，用固定目标检查连续更新。无需启动 OpenPI 服务。
每次从初始末端位姿生成一个目标，随后每步重新计算当前位姿到同一个目标的误差。
位置与姿态均参与反馈，避免另一部分自由漂移。全程保持夹爪张开，不接触方块。

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test track-x
```

`track-x` / `track-y` / `track-z` 的固定目标为对应根坐标轴正向 5 mm；
`track-rx` / `track-ry` / `track-rz` 的固定目标为对应根坐标轴正向 2°。
每步相对 IK 指令的平移向量范数最多 1 mm，旋转向量范数最多 0.5°，再按 IK scale 换算。
各测试执行 120 个控制步，超过实际旋转 15°、位移 5 cm，或环境重置时停止。

输出为 `logs/openpi_camera_check/motion_track-x.json` 等文件。
终端只显示摘要，完整文件包含逐步 `trace`：指令、实际位姿、时间和目标误差。
`reached_and_held_last_5_steps` 仅在最后 5 步均满足位置误差不超过 0.25 mm、
旋转误差不超过 0.1° 时为 `true`。未达标也会保存轨迹，便于检查停滞、反向或振荡。

建议先运行 `track-x` 与 `track-rx`，查看摘要及轨迹，再扩展到其余四轴。
此测试确认固定目标反馈是否收敛；不能据此把 OpenPI 原始动作解释为米或弧度。
本地 CPU 检查覆盖限幅、符号和简化伺服收敛，不替代 Spark 上的真实仿真测试。

## LIBERO 控制器动作映射验证

六轴固定目标都通过后，运行一次自动映射测试：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test libero-map
```

不需要 OpenPI 服务。自动执行六轴各一次正向与反向指令，再张开、闭合、重新张开。
每个平移目标为 5 mm、旋转目标为 2°。测试使用以下明确的参考配置：

- LIBERO 默认 `OSC_POSE`、`control_freq=20`。
- robosuite 1.4.0 默认输入范围 [-1, 1]；平移乘 0.05 m，旋转乘 0.5 rad。
- 例如控制器动作 x=0.1 对应 5 mm，rx=0.06981317 对应约 2°。
- Panda 夹爪动作正值闭合、负值张开；适配为 Isaac 正值张开、负值闭合。
- 零夹爪动作沿用上一个二值指令；二值适配不复现 robosuite 的手指渐进驱动。
- 旋转为世界坐标系左乘增量。世界平移/旋转向量先按实际机器人根姿态变换到根坐标系。

脚本只在 `libero-map` 模式把控制周期设为 0.05 s，保留物理步长，要求能用整数 decimation 表示。
每个固定目标持续反馈 60 步（3 秒仿真时间），完整测试约 45 秒仿真时间。
这是映射后的目标跟踪测试，**不是每个模型动作都执行 3 秒**，也不表示 IK 与 OSC 动态等价。
完整轨迹中的 `first_period_response` 单独记录首个 50 ms 的响应，供后续时序比较。
每次重置即退出，不继续使用旧目标。仍有实际位移 5 cm、旋转 15° 的中止条件。

终端每阶段显示 `target=True, gripper=True`，最终 `all_mapping_targets_passed` 应为 `true`。
输出 `logs/openpi_camera_check/motion_libero-map.json` 包含 15 个阶段、输入动作、映射后的目标、
首周期响应、最终误差、手指位置和逐步轨迹。达标阈值沿用位置 0.25 mm、旋转 0.1°；
手指位置应在张开 0.04 m 或闭合 0 m 的 0.002 m 范围内。

此配置来自公开默认值，并未读取用户本机 LIBERO 配置。若本机修改过控制器或频率，需先对齐配置。
当前假设两个场景的世界坐标轴对齐，尚未完成跨场景配准、末端参考点、相机外参及观测状态校准。
因此通过测试也不代表 pi05_libero 可以零样本抓取。尤其当前 Isaac 两指关节值同为正数，
不能仅凭状态为 8 维就认为已经匹配 LIBERO 的两指关节符号与范围。

原有 `--query` 现在额外保存 `predicted_action_mapping.json`，展示模型输出的控制器映射与被裁剪的维度；
仍然只保存，不执行模型动作。首次设置二值夹爪为张开，再按预测动作更新。
不要对已经经过 OpenPI 服务输出变换的动作再次应用训练归一化统计。

参考源码：

- [LIBERO 环境默认控制器和频率](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)
- [LIBERO 依赖版本](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt)
- [robosuite 1.4.0 OSC_POSE 缩放](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json)
- [robosuite 1.4.0 旋转目标](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/control_utils.py)
- [Panda 夹爪方向](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)
- [OpenPI LIBERO 官方执行示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)

本地运行映射 CPU 检查：`python3 openpi_isaac_bridge/test_libero_mapping.py`。

## 连续动作时序测试

`libero-map` 全部通过后，运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test libero-stream
```

无需 OpenPI 服务。对六轴逐一执行：正向 20 步、零动作 10 步、反向 20 步、零动作 20 步。
每个输入只占一个 50 ms 仿真周期，周期内不额外等待目标收敛。
每步目标基于当前实测末端位姿生成，不把增量加在上一期的期望目标上。
平移每步目标 0.25 mm，20 步名义行程 5 mm；旋转每步目标 0.1°，名义行程 2°。
这里的总行程是理想累计参考，实际响应由仿真决定，不能用固定目标测试的收敛结果替代。
夹爪全程张开，六轴共 420 个控制步、21 秒仿真时间；墙钟时间可能更长。

终端每轴显示 `passed=True`，最终 `all_stream_tests_passed` 应为 `true`。
结果文件 `logs/openpi_camera_check/motion_libero-stream.json` 保存每步输入、目标、实际位姿及误差。
关键指标：

- `outbound_gain`：正向阶段实际位移/转角在名义方向上的投影比，诊断范围 0.7～1.3。
- `max_period_*` / `rms_period_*`：当期目标与期末实际位姿的最大/RMS 误差，用于观察动态跟踪，不能直接解释为网络延迟。
- `return_position_error_m` / `return_rotation_error_deg`：往返后的闭合误差，阈值 1 mm / 0.3°。
- `stop_drifts`：每次输入归零后的位置/姿态变化，阈值 0.5 mm / 0.2°。
- `mean_wall_period_s`：包括控制、渲染和主动等待的平均墙钟周期；0.05 s 仿真步长不保证真实时间 20 Hz。

最大单周期位置误差阈值为 1 mm、姿态误差为 0.5°，夹爪还应保持张开。
这些是当前小幅诊断的工程判据，不是官方模型评测标准，也不证明 OSC 动态等价。
任一轴未通过仍会保存完整轨迹，优先检查目标是否连续更新、实际响应比例、停止漂移与周期耗时。
测试不调用模型，不覆盖异步推理、动作块队列或真实网络延迟；后续仍需单独验证。

## 阶跃响应及控制点核对

连续输入的实际行程偏小时，先运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab -p \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/check_cameras.py \
  --viz viser --motion-test step-response
```

无需 OpenPI 服务，全程保持夹爪张开，不改执行器增益，不放大模型动作。
固定目标为 X 轴 +1 mm 或 RX +0.5°，每个目标连续保持 60 个 50 ms 周期（3 秒）。
分别用 `sensor`（现有 ee_frame 测量点）和 `ik`（动作项实际控制点）做反馈，共四项比较。
各项结束后用 60 步返回该项起点，再进行下一项，总计约 24 秒仿真时间。
每一步重算到同一个固定目标的误差，不累加期望目标。

结果保存为 `logs/openpi_camera_check/motion_step-response.json`。终端包含：

- `frame_alignment`：IK 的 body_name/body_offset、传感器目标 prim/offset、两个实测点之间的位置及姿态差。
- 四项 `[Step response]`：50、100、250、500、1000、3000 ms 的行程比例 `gain`。
- `t90_sustained_s`：达到 90% 且在剩余观测时间内始终不低于 90% 的首次采样时间。
- `settling_5pct_s`：进入目标幅度 ±5% 且保持至观测结束的首次采样时间；`null` 表示 3 秒内未满足。
- `checkpoints`：上述时间点的幅度、完整位姿误差，以及控制器内部目标相对固定目标的差。

完整 `trace` 还记录每步指令、传感器位姿、IK 位姿和控制器内部期望位姿。
`sensor` 模式中，传感器点和 IK 点偏移会反映在内部目标与传感器固定目标的差中；
两者不是同一物理点时，该差本身不等于错误。应结合 `ik` 对照、随时间变化情况和真实跟踪误差判断。
测试只读取私有 `_compute_frame_pose` 与 `_ik_controller` 诊断字段，不写这些字段；
若安装版本缺少这些接口，会明确停止，不采用猜测控制点的替代结果。

这些是当前 IK/伺服闭环的阶跃响应，不是单独电机性能，也不能据此推断 OSC、网络或模型推理延迟。
先查看 50 ms 响应是否约为 20%，后续是否逐步趋近 100%，再比较两个反馈点的差异。
需要 CPU 检查摘要计算时运行：`python3 openpi_isaac_bridge/test_step_response.py`。

## 执行器增益对照测试

若阶跃在 50 ms 仅完成约 20%、长时间保持后却能到达目标，可用本轮对照检验执行器增益是否影响响应。
这仍是假设检验，不预先认定刚度/阻尼就是唯一原因。
三组分别为 `baseline`（原参数）、`kp1p5`（刚度 ×1.5，阻尼保持）、
`kp1p5-damped`（刚度 ×1.5、阻尼 ×√1.5）。后者只是一个对照选择，不保证保持实际关节阻尼比。
每组重新创建场景；只调整覆盖七个 Panda 机械臂关节的隐式执行器，手指、速度限制、力矩限制、IK 参数和物理步长不变。
若增益由 USD 继承、配置混合机械臂和手指，或执行器不是隐式模型，会明确停止。

一次执行三组的阶跃与连续测试（共六次独立仿真进程）：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab
```

这个外层程序只用 Python 标准库，不直接加载 Isaac Sim；每个子进程仍通过
`uv run --extra viser isaaclab -p ...` 启动，激活源码版 Isaac Sim 环境。
运行前结束已有测试进程，避免 Viser 端口冲突。无需 OpenPI 服务。
每组总计约 45 秒仿真时间，墙钟时间还包括渲染与六次启动。

输出写入独立时间戳目录 `/home/test/Robot/IsaacLab/logs/openpi_gain_sweep/<UTC时间戳>/`，
不会覆盖前一轮轨迹。每个配置、每种测试各自保存：

- `console.log`：完整日志；外层终端只转发关键摘要。
- `actuator_audit.json`：原始/实验配置、实际关节刚度/阻尼、速度/力矩限制及 IK 配置。
- `joint_telemetry.json`：逐控制周期的关节速度与可用的力矩遥测。
- `motion_step-response.json` 或 `motion_libero-stream.json`：完整轨迹和性能指标。

外层每次运行后更新汇总：

```bash
cat /home/test/Robot/IsaacLab/logs/openpi_gain_sweep/latest_gain_sweep.json
```

`status=completed` 只表示六个进程完成，不表示运动测试全部通过；
看 `responses` 中的 `t90_sustained_s`、`settling_5pct_s`、`overshoot_pct`、`tail_gain_peak_to_peak`，
并比较连续测试各轴的 `outbound_gain`、返回误差和停止漂移。
`joint_telemetry_summary` 给出逐关节采样速度/估算力矩的绝对峰值及其与已知限制的比例。
隐式执行器力矩不是独立测力传感器读数；每 50 ms 采样也可能漏掉物理步内峰值和振荡。
未提供的遥测或限制明确标为 unavailable/null，不能据此声称未触及限制。
后 10 个采样的幅度峰峰值只是尾部变化指标，不单独证明存在或不存在振荡。

若子进程报错，外层保存已完成的结果，显示日志尾部并停止；运动指标不达标则继续完成对照。
脚本不会自动选择或永久应用某组参数，不修改 USD 或官方环境源码。
如需单独运行某项，可使用原命令加 `--gain-profile kp1p5` 与独立 `--output` 目录；
实验增益只允许用于 `step-response` / `libero-stream`。

本地 CPU 检查：`python3 openpi_isaac_bridge/test_gain_probe.py`。

## 第二轮：固定刚度，降低阻尼

首轮实际加载的机械臂参数为刚度 400、阻尼 80。固定刚度 600，对照阻尼 80、60、40：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp1p5 kp1p5-kd0p75 kp1p5-kd0p5
```

参数仍按新建环境的原始增益缩放：三组均为原始刚度 ×1.5，阻尼分别为 ×1、×0.75、×0.5。
若本机原始配置已改动，实际值可能不是 600/80、600/60、600/40，以 `actuator_audit.json` 为准。
每组从新进程重新创建环境，不在前一组参数上继续相乘。
不传 `--profiles` 时仍保持首轮三组默认行为，不会自动运行新增的全部配置。

本轮仍运行阶跃与六轴连续测试，保持手指参数及速度/力矩限制不变，保存独立时间戳目录。
重点比较：50 ms 目标完成比例、达到90%的时间、超调、尾部变化、停止漂移、以及采样关节限制比例。
降低阻尼可能加快响应，也可能增加超调；不能仅选 `outbound_gain` 最大的一组。
原有 5 cm / 15° 位姿中止条件继续生效；它不是物理步级振荡检测器。
本轮完成后，将新汇总与各组 `actuator_audit.json` 上传到 `diagnostics/gain_sweep/`，再进行对比分析。

## 第三轮：50 ms 完成比例验证

保留已测试的 600/40，逐步对照 600/30、600/20、900/20、1200/20：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp1p5-kd0p5 kp1p5-kd0p375 kp1p5-kd0p25 kp2p25-kd0p25 kp3-kd0p25
```

五组各运行阶跃和六轴连续测试，共十次独立仿真，总计约 225 秒仿真时间，另加启动及渲染耗时。
上表数值以原始刚度400、阻尼80为前提；参数是相对于每次新建环境原配置的倍率，
不是在上一组上叠加。实际加载值仍以 `actuator_audit.json` 核对。
默认三组旧测试不变；本轮必须显式传 `--profiles`。

阶跃与外层汇总新增：

- `gain_at_50ms`：固定目标在首个 50 ms 后的实际幅度比例。
- `within_10pct_at_50ms`：比例处于 [0.9, 1.1]。
- `within_5pct_at_50ms`：比例处于 [0.95, 1.05]。

这两个布尔值只表示被测轴的首周期幅度误差，不是完整位姿、连续运动或后续稳定性达标。
仍需查看完整位置/姿态误差、`overshoot_pct`、`settling_5pct_s`、尾部变化、六轴行程比例、停止漂移及限制比例。
例如首周期比例1.2虽然超过90%，也属于超调，不通过上述幅度范围判据。
源码、USD、手指驱动、速度/力矩限制、物理步长不变，不执行模型动作。
这些配置仅是待验证候选，不能保证达到目标；仿真仍同步运行，50 ms是仿真周期，不保证墙钟实时性。

## 第四轮：以900/20为对照，抑制超调

第三轮900/20的连续行程较接近名义值，但平移阶跃最大采样超调约21.6%。
固定刚度900，比较阻尼20、30、40，检查增大阻尼能否降低超调，以及响应会慢多少：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp2p25-kd0p25 kp2p25-kd0p375 kp2p25-kd0p5
```

三组从原始配置400/80分别乘以刚度2.25、阻尼0.25/0.375/0.5。
若本机原配置不同，以实际加载参数审计为准。默认三组测试不变，本轮显式选择以上配置。
仍为六次独立仿真：每组阶跃与六轴连续测试各一次，不更改USD、执行器限制或手指参数。

本轮不预设提高阻尼后仍能满足50 ms指标，也不将降低超调等同于已经足够快。
重点同时查看 `gain_at_50ms`、`overshoot_pct`、`settling_5pct_s`、连续行程比例和停止漂移。
例如，可把首周期幅度在90%～110%、全程采样超调不超过5%作为待检验的组合目标；
它是本项目的诊断判据，不是LIBERO官方标准，也不代表所有轴或接触抓取已通过。
当前阶跃仅验证X与RX，且采样周期50 ms，可能漏掉周期内峰值；不能据此声称所有关节都接近临界阻尼。
若三组不能兼顾速度与超调，应考虑按关节整定或控制接口改进，不继续只按首周期幅度选配置。

## 每个物理子步记录七关节目标与响应

先采集900/20、900/30、900/40的高分辨率轨迹，不自动改变各关节增益：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp2p25-kd0p25 kp2p25-kd0p375 kp2p25-kd0p5 \
  --record-physics
```

动作周期仍为50 ms，物理步长不变；记录频率等于现有物理频率。
例如物理步长为10 ms，每个动作应记录5条；若为5 ms，应记录10条，以启动日志为准。
采样只增加读取和文件写入，不额外推进仿真，不以 `sleep` 制造数据点。
同步读取GPU数据会增加墙钟耗时，这是诊断功能。

记录器在场景 `write_data_to_sim` 后捕获此次物理积分实际使用的关节目标，
在随后 `scene.update` 后捕获响应。IK可能在同一个50 ms动作周期内多次更新关节目标，
所以不能简单拿动作周期结束时的目标与所有子步比较。
记录器只在测试动作期间启用，恢复原场景方法后再关闭环境；不记录预热和相机初始化。
若后端内部一次处理整个decimation、每周期没有完整子步、或更新dt不匹配，会明确停止，
不把50 ms的批量状态冒充更高频率的数据。

每个测试目录新增：

- `physics_joint_trace.jsonl`：逐物理步的七关节实际目标、积分前位置、积分后位置/速度、可用的估算力矩。
- `physics_joint_summary.json`：按case/phase、关节分别计算RMS/最大跟踪误差、目标跨度、速度及估算力矩峰值。

`command_time_s`与`time_s`分别是该步起止时刻，时间原点是第一个记录的诊断动作。
`control_index`、`physics_substep`和case/phase标签区分控制周期、子步、阶跃/返回/连续/停止。
估算力矩在写命令后、积分前读取，不是求解器真实力矩；物理采样仍可能漏掉更高频的现象。
`normalized_rms_error`以该case内目标相对初始位置的最大变化归一化，变化小于1e-5 rad时为null，
避免把几乎未激励的关节噪声误判为慢响应。
这些统计是IK运行中的联合跟踪结果，不是各关节独立阶跃辨识；仅凭它们不能直接估计每关节阻尼比。

汇总自动包含关节统计。运行结束后，可用通用导出程序准备上传，避免每轮手改配置检查：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add physics substep joint target and response traces"
git push origin main
```

导出程序先确认最新测试完成并读取全部必需文件，再更新仓库文件。
它复制结果汇总、实际执行器参数、物理采样汇总和gzip压缩后的完整轨迹，不自动提交或推送。
路径为 `diagnostics/gain_sweep/<profile>_<mode>_physics_joint_trace.jsonl.gz`。
旧轮次的其他配置文件不会删除，分析时以新 `summary.json` 列出的本轮runs为准。

### 数据分析后再按关节修改增益

已增加 `--joint-gains-json`，仅允许指定 `panda_joint1` 至 `panda_joint7` 的正数stiffness/damping。
未指定的关节/增益沿用所选profile，手指与速度/力矩限制保持不变。
例如下面仅演示文件格式，不是已经验证的调参建议：

```json
{"panda_joint2": {"damping": 35}}
```

如将该文件保存为 `/home/test/Robot/joint_gains.json`，可在采样命令后追加
`--joint-gains-json /home/test/Robot/joint_gains.json`。
覆盖先应用profile，再应用逐关节值；最终每个关节实际增益仍由audit记录。
首轮高分辨率采集不添加该参数，先确认哪些关节响应或目标更新异常，再选具体覆盖值。

## 首个数据驱动的逐关节对照候选

已上传的首轮物理轨迹（提交 `b6d2c1399a68e6787a2b56696430391cf5920289`）确认物理步长10 ms，
每个50 ms动作周期有5条记录，共13500条物理样本。
X向测试主要激励关节2、4、6。以各关节最终命令位置为参考，900/20时关节2、4分别在约100/90 ms
出现约21.4%/27.3%的峰值越过终值；关节6约4.9%。
这些数值是联合运动中的终值参考指标，不是独立关节阶跃的严格超调辨识。
RX主要激励关节1、3、5、7，其目标也在随IK更新，不能把它们当作固定关节目标来直接拟合阻尼比。

新增 `joint_gains_j2_j4_d40.json`：以900/20为基础，仅把关节2、4阻尼改为40。
其它五个机械臂关节保持900/20，手指与执行器限制不变。
这是尚未仿真验证的最小对照候选，不承诺达到50 ms或代表最终整定参数。
它检验能否降低X向超调，同时减少统一提高阻尼对其它方向响应的影响。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp2p25-kd0p25 \
  --joint-gains-json /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/joint_gains_j2_j4_d40.json \
  --record-physics
```

本轮只需要一个profile、两次测试。汇总显式保存逐关节覆盖值与各次运行的实际执行器审计，
不能仅凭profile名称把这组误认作七个关节都是900/20。
运行后仍用 `export_gain_results.py --include-physics-traces` 导出，再提交结果。

另一个关键修正：物理采样显示首轮阶跃的估算力矩峰值可达约48.2%求解器上限，
而先前50 ms采样最高约5.4%。因此不能沿用“力矩一直很小”的判断。
这些仍是隐式执行器估算值而非真实测力，当前数据未显示触顶，后续保留原限制继续观察。
