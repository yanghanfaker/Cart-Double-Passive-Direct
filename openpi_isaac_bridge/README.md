# Isaac Lab / OpenPI 相机与观测检查

用于已能运行 `IsaacContrib-Lift-Cube-Franka-IK-Rel` 的 Isaac Lab 3 开发版本。
脚本默认采集两台相机和8维状态；`--query`只保存模型输出。
只有显式选择 `--motion-test openpi-step` 才执行限幅的模型末端动作，并保持夹爪打开；见文末首轮模型测试说明。
它尚未完成OpenPI视觉闭环抓取或LIBERO与Isaac场景配准。

用户已在Spark完成图片采集、模型查询、动作映射、关节/末端回归和真值定位的确定性抓取；各轮证据见后文。
新OpenPI限幅执行模式目前通过CPU检查，实际模型闭环仍待Spark验证。
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

## 下一轮：关节2、4、6固定关节目标阶跃

逐关节候选的实测结果（提交 `143b8600f65c9153fffd1895f1af8219e3880c6e`）显示：
仅提高关节2、4阻尼后，X方向超调降低，但50 ms响应降至约67%，连续Z向增益约0.507。
这组候选不能作为最终参数。下一轮直接测关节响应，进一步区分执行器动态与IK、多关节耦合。

`joint-step` 模式绕过机械臂IK动作的最终应用，直接写入七个关节的位置目标和零速度目标。
依次对 `panda_joint2/4/6` 各施加正、负0.003 rad（约0.172度）阶跃；
其余机械臂关节的位置目标保持起始值，夹爪保持打开。
其它关节仍由执行器保持位置，实际位置可以因耦合而变化，并非机械锁死。
每个方向保持3秒、回到起点3秒；每组配置约37秒仿真时间，另加启动和渲染耗时。

本轮比较七关节统一Kp=900、Kd=20/30/40三个配置，命令不添加之前的逐关节覆盖文件：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --profiles kp2p25-kd0p25 kp2p25-kd0p375 kp2p25-kd0p5 \
  --modes joint-step \
  --record-physics
```

该模式自动启用物理采样，在每次物理写入后检查七个目标是否与固定命令一致（容差1e-6 rad）；
目标被IK覆盖、越过软限位、回位误差超过0.0003 rad，均停止本轮。
控制周期仍为50 ms，沿用物理步长逐步采样；当前环境实测10 ms，不通过额外推进仿真来增加样本。
本轮不会请求或执行OpenPI动作。

每组包含六个阶跃，输出 `[Joint step]` 日志与 `motion_joint-step.json`，
记录50 ms位移/名义阶跃幅度、峰值、超调、持续达到90%的时间、5%调节时间、
最终目标误差和其它关节最大位置偏移（`coupling_peak_rad`）。
完整物理轨迹、执行器参数审计同时保存；正负响应差异及起始静态误差需结合原始轨迹解释。
这些仍是当前姿态下的耦合机械系统响应，不能直接当作准确惯量或极点辨识。

脚本的CPU检查不替代Spark上的Isaac运行验证。若运行报错，保留本轮console.log用于定位。
完成后使用同一导出入口上传结果（现在支持只运行一个测试模式）：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add isolated joint 2 4 6 step response results"
git push origin main
```

## 关节2、4联合调整刚度和阻尼

固定关节阶跃结果（提交 `47579c0c92e20e8f875871880a04b6a78b31aa58`）已完成三组对照：
3700条物理样本/配置，共11100条；逐步核对固定位置目标和零速度目标。
900/20时关节2/4的50 ms响应约63.4%/69.9%，超调约28.0%/23.6%；
900/40时50 ms响应仅49.6%/53.3%，超调仍约9.5%/6.9%。
关节6在900/20时50 ms达到93.5%，超调约0.7%。
关节2与4之间还存在约阶跃幅度21%的非激励关节瞬态偏移，因此这些不是完全解耦的对象。

下一轮使用 `joint_gain_cases.json`，只覆盖关节2、4，其余五个机械臂关节均保持900/20：

|case|关节2/4 Kp|关节2/4 Kd|用途|
|---|---:|---:|---|
|control-900-20|900|20|复测基准|
|j24-kp900-kd40|900|40|逐关节高阻尼对照|
|j24-kp1800-kd56p57|1800|56.5685|从900/40沿近似等阻尼比路线提高带宽|
|j24-kp3600-kd80|3600|80|同路线更高刚度|
|j24-kp1800-kd42p43|1800|42.4264|从900/30沿较低阻尼比路线对照|
|j24-kp3600-kd60|3600|60|同路线更高刚度|

路线依据是局部二阶近似：Kp乘以a，Kd乘以sqrt(a)，有望在提高自然频率时近似保持阻尼比。
实际系统有多关节耦合、隐式驱动和离散求解，这不是已辨识参数，不保证等阻尼或50 ms达标。
本轮不改变物理步长、控制周期、速度限制、力矩限制和手指参数。
关节2/4最大刚度3600下，0.003 rad阶跃对应初始纯比例项约10.8 Nm；实际估算力矩和响应仍以轨迹为准。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/joint_gain_cases.json \
  --modes joint-step \
  --record-physics
```

共六个独立仿真进程；每组约37秒仿真时间，启动、渲染及日志读回另外耗时。
仍测关节2/4/6的正负0.003 rad阶跃，关节6用于检查参数变化引发的耦合影响。
输入case名称用于文件夹和导出前缀；真实base_profile、每关节覆盖值、启动命令及实际审计均存入汇总，
避免不同覆盖值共用同一基础profile名称而覆盖结果。
`--gain-cases-json`不能与`--profiles`或`--joint-gains-json`同时指定。

先比较50 ms增益是否达到0.9至1.1、超调是否不超过5%，同时检查调节时间、回位、耦合和估算力矩。
这些是本项目筛选目标，不是模型官方标准。没有候选满足时保留失败数据，不自动选参数或扩大动作。
即使单关节测试达标，仍须再测末端XYZ/RXYZ连续动作及其它姿态，才能判断实际使用效果。
修正了physics summary的scope：现在直接使用本轮target_note，不再给固定关节测试标注旧的IK目标说明。

运行完成后：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add joint 2 4 stiffness damping sweep results"
git push origin main
```

本轮脚本仅完成CPU配置、路由及统计检查，新的增益候选尚待Spark仿真验证。

## 围绕3600/80的窄范围细调

上一轮六组结果（提交 `4c4813664afc3a2c73284ef38d7105d4f953e660`）共22200条物理样本，
实际增益与固定目标复核通过。关节2/4在3600/80时50 ms行程约94.1%/94.1%，
但约80 ms出现峰值，超调约9.1%/6.6%，5%调节时间约120/110 ms。
3600/60虽然50 ms约105%，后续超调却约16.3%/12.7%，不能据50 ms一个点选为最终参数。

新输入文件 `joint_gain_refine_cases.json` 保留上一轮文件不变，只覆盖关节2、4：

|关节2/4 Kp|Kd|作用|
|---:|---:|---|
|3600|80|复测基准|
|3600|90|固定刚度，观察增加阻尼的速度/超调取舍|
|3600|100|同刚度更高阻尼对照|
|4500|100|联合提高刚度与阻尼|
|4500|110|同刚度更高阻尼对照|
|5400|110|继续提高刚度以补偿阻尼带来的行程损失|
|5400|120|同刚度更高阻尼对照|

其余五个机械臂关节为900/20，手指和力矩/速度上限保持原配置。
这些是实验候选，不是已预测或保证的响应结果。控制仍50 ms，物理步长沿用当前环境的10 ms。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/joint_gain_refine_cases.json \
  --modes joint-step \
  --record-physics
```

共七组，每组仍测关节2/4/6正负0.003 rad阶跃及回位，约37秒仿真时间/组。
每个候选独立进程启动，继续核对固定目标、软限位、回位误差和实际参数。

新增峰值时刻 `peak_time_s`，以及汇总中的 `joint_step_screen`。
`[Joint screen] ... complete=True, all_passed=True` 仅表示六次阶跃都具备有限数值，
50 ms行程比例在[0.9,1.1]、全程超调在[0,5]%范围内。
遗漏方向、重复记录、关节6退化也不能通过该初筛。它不会自动选用或部署参数；
没有通过时测试仍正常保存完成结果，不能把completed和all_passed混为一谈。
即使通过，还需核对调节时间、耦合、估算力矩及其它姿态，然后回归末端XYZ/RXYZ连续动作。

运行后推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add refined joint 2 4 gain sweep results"
git push origin main
```

## 两个通过关节初筛的候选：末端连续动作回归

窄范围细调结果（提交 `71e995db40ebea7699faec797b4b91433a9745b7`）中，
关节2/4的4500/110与5400/120通过关节2/4/6正负小阶跃初筛。
5400/120的关节2/4 50 ms行程约95.3%/94.4%，超调约4.3%/2.9%；
其中关节4进入并保持±5%仍需60 ms。此结论不代表末端连续控制或抓取已经通过。

下一轮使用 `cartesian_regression_cases.json`，独立比较三组：

|case|关节2/4参数|其它五个机械臂关节|
|---|---|---|
|cart-control-900-20|900/20|900/20|
|cart-j24-kp4500-kd110|4500/110|900/20|
|cart-j24-kp5400-kd120|5400/120|900/20|

这里恢复正常IK机械臂动作应用，单环境、夹爪打开，仍不请求或执行OpenPI模型输出。
保留原物理步长、50 ms控制周期、速度/力矩限制。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/cartesian_regression_cases.json \
  --modes step-response libero-stream \
  --record-physics
```

三个配置各运行两个模式，共六个独立仿真进程：
- `step-response`：已有sensor/IK两种反馈下的X、RX固定末端目标阶跃与回位，观察50 ms响应、峰值和调节时间。
- `libero-stream`：XYZ/RXYZ六方向，每轴20个动作向外、10个零动作停止、20个反向动作、20个零动作停止。
  每50 ms仅执行一次基于当前实测姿态的相对目标，不累计期望位姿，不插入额外收敛等待。
  平移单次0.25 mm，20次名义累计5 mm；旋转单次0.1度，20次名义累计2度。

完整逐周期目标与实际末端位姿现随汇总导出，物理步仍记录七关节命令/响应和估算力矩。
这是初始姿态附近的小幅回归，后续仍须扩展姿态、幅度和负载。

原 `all_stream_tests_passed` 的增益范围[0.7,1.3]保留，便于与历史数据对照。
新增 `cartesian_stream_screen`，要求六方向齐全、全部原检查通过，并把向外段累计增益限制为[0.9,1.1]。
原检查还包括：回位位置≤1 mm、回位角度≤0.3度；每周期末目标误差≤1 mm/0.5度；
停止段位移≤0.5 mm、转角≤0.2度；两手指位置接近打开目标。
`[Cartesian screen] ... all_passed=True`仅代表上述小幅测试通过。
累计增益是20次动作的整段投影位移比例，不等同单次50 ms响应、每周期±10%精度或超调指标。
不能把测试通过解释为LIBERO OSC动力学等价、视觉/状态标定完成或OpenPI抓取成功。

若某方向失败，正常保存结果供分析；completed表示流程完成，不表示指标通过。
运行完成后推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add Cartesian regression for tuned joint gains"
git push origin main
```

## 扩展起始末端位置和连续动作幅度

初始位置下的末端回归结果（提交 `ecea848404f13259548f5aa798b739b468a452c5`）三组均通过小幅连续动作标准。
4500/110与5400/120显著减小X固定目标超调及X/Z/RY停止漂移，但没有所有方向全面占优的候选。
下一轮保持这两个候选与原900/20对照，扩展局部起始位置和动作幅度。

`cartesian_workspace_scenarios.json` 包含三处起点和两档幅度：

|起始位置|相对原始末端位置的偏移（机器人根坐标，米）|末端朝向|
|---|---|---|
|home|[0, 0, 0]|保持原朝向|
|yplus|[0, 0.015, 0.010]|保持原朝向|
|yminus|[0, -0.015, 0.010]|保持原朝向|

这会改变机械臂关节构型，但尚未覆盖不同末端朝向、大范围工作空间或负载变化。
幅度1为原测试：每周期0.25 mm或0.1度，20次名义累计5 mm/2度；
幅度2为每周期0.5 mm或0.2度，20次名义累计10 mm/4度。
两档均维持50 ms控制周期和out20/stop10/back20/stop20的70步序列。
夹爪始终打开，不执行OpenPI输出。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/cartesian_regression_cases.json \
  --stream-scenarios-json openpi_isaac_bridge/cartesian_workspace_scenarios.json \
  --modes libero-stream \
  --record-physics
```

总计3组参数×3起点×2幅度=18个独立进程，各场景重新创建环境。
场景名称、base_case、起点偏移和幅度显式写入汇总，输出不会覆盖。
增加场景选项时只允许运行libero-stream，避免把这些条件误用于其它模式。

每个场景先用固定末端位姿闭环控制80个周期（4秒仿真时间），保持初始末端朝向。
最后连续5步位置误差≤0.25 mm且角度误差≤0.1度才开始六轴序列；未到位则停止本轮。
包含home场景也执行相同预定位等待，以统一条件。
`stream_start` 保存起点目标、实际位置、朝向、七关节位置和到位误差轨迹。
物理记录中的 `case=stream-preposition` 是预定位数据，不属于轴向响应。
原5 cm位移和15度姿态保护仍生效，预定位不会重置这个参考范围。

每场景500控制步、当前10 ms物理步长下2500物理样本；18场景共45000条物理样本。
每场景25秒仿真时间，总450秒仿真时间，另加18次启动、渲染和日志读回耗时，不保证墙钟运行时间。
原有累计增益、回位、周期目标误差、停止漂移的阈值保持不变，不随幅度放宽。
`[Cartesian screen]`仍检查六轴累计增益90%至110%及原有其它条件。

观察候选是否在不同起点/幅度仍通过，以及是否出现方向性行程变化、耦合、停止漂移或估算力矩升高。
流程completed与指标all_passed分开：指标不通过仍保存结果；预定位或仿真异常会停止并保留console.log。
新的起点控制尚待Spark实测，CPU检查不替代运行验证。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add workspace and amplitude Cartesian regression results"
git push origin main
```

## 下一阶段：确定性接触、闭爪与抬升诊断

位置/幅度回归结果（提交 `d27d7e5e8a73587db85ef8cd334815739bfaab71`）中，
4500/110和5400/120各自6个局部场景均通过；这不是抓取成功率，也未验证接触。
新增 `grasp` 模式与 `grasp_validation_cases.json`，按5400/120、4500/110、原900/20顺序对照。
只覆盖关节2、4的增益，其它机械臂关节、手指及速度/力矩限制保持原配置。

本轮是使用仿真真值的控制/几何诊断，不是OpenPI推理、视觉定位或最终抓取策略。
固定物体reset位置偏移为零，seed=0；保留任务原始物体USD、缩放和物理属性。
原任务默认物体和传感器设置见官方
[joint_pos_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka/joint_pos_env_cfg.py)，
随机reset设置见
[lift_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/lift/lift_env_cfg.py)。
本地实际配置以输出 `grasp_setup` 为准，它记录物体资产、缩放、reset设置、机器人重力开关和两个TCP偏移。
末端传感器默认offset为0.1034 m，IK控制点默认0.107 m；本轮统一用传感器闭环反馈，不声称偏移已标定等价。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/grasp_validation_cases.json \
  --modes grasp \
  --record-physics
```

三组分别创建一个环境，执行一次以下流程：
1. 夹爪打开，在初始较高位置将末端转到Rx(pi)朝下姿态（根坐标xyzw=[1,0,0,0]）。
2. 移到物体初始中心正上方100 mm。
3. 下降到物体初始中心上方5 mm的TCP目标。
4. 固定末端目标，闭爪80周期（4秒），检查最后10周期的两指位置稳定性及物体是否在TCP附近。
5. 只有闭爪候选检查通过才上提50 mm，随后保持40周期（2秒）。

每个移动阶段须连续5周期误差≤2 mm、朝向误差≤2度才进入下一阶段，并设最大步数超时。
闭爪候选要求两指各大于2 mm、合计开度6–70 mm、两指差≤4 mm，最后10周期各指位置波动≤0.5 mm，
物体中心距实际TCP≤15 mm，且末端目标误差保持容差内。
这些条件只表示受阻夹持候选，不能替代接触力测量；夹爪空闭会跳过抬升。

成功须在整个2秒保持期物体相对初始高度至少增加35 mm，
物体相对末端的位置漂移≤10 mm、角度漂移≤10度，
从闭爪结束到抬升/保持期间的相对位置变化≤15 mm，且夹爪与末端保持条件成立。
每周期记录末端、物体根坐标位姿、物体在末端坐标的相对位姿、两指位置及目标误差。
物体速度由50 ms位置差估算，不是高频测得的速度或接触力；估算速度超过0.5 m/s、
抬升前物体移动超过20 mm或低于初始支撑高度10 mm，会中止本次诊断并记录原因。

TCP目标位于中心上方5 mm以及固定朝下姿态是首轮几何假设，尚未仿真验证；
如果夹不住，先看末端与物体相对位置及手指开度，不能直接归因于增益不够。
本轮不加抓取辅助约束、不把物体绑定到夹爪、不修改摩擦/质量、不执行VLA输出。

抓取需要比前期5 cm/15度更大的接近和转向范围，因此仅此模式使用0.6 m整体位移范围；
仍将每50 ms的IK目标增量限制为1 mm、0.5度，并检查实际单周期位移≤10 mm、旋转≤3度及关节软限位。
控制周期50 ms，物理步长沿用当前环境10 ms。

`[Grasp]`依次打印阶段到位、闭爪候选和最终success。
`grasp_result.success=False`表示测量未达标，仍保存完整trace、failure_reason和实际增益，流程可completed；
环境重置、非预期API或保护检查异常则使运行失败，保留console.log和已写入的物理轨迹。
CPU测试中的合成状态仅验证流程和判定，不代表Isaac接触物理已通过。

运行完成后推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add deterministic grasp contact and lift diagnostics"
git push origin main
```

## 改变物体位置/偏航、重复抓取与慢转向对照

首轮接触诊断（提交 `f1c1126ab411f5e2831b6ccb3e9a3af04ef5aff0`）中两个候选均夹稳并抬升约49.8 mm、保持2秒。
原900/20夹爪仍夹着物体，但持物跟踪偏移并在抬升阶段超时。
另一个独立发现：三组在高处转向阶段，关节6估算力矩均出现达到12 Nm原上限的样本。
这轮保留两个候选，扩展物体位置/偏航并比较慢转向，不改变手指、力矩/速度上限或成功阈值。

使用 `grasp_robustness_cases.json` 的5400/120、4500/110两组，与以下四种条件交叉：

|条件|相对物体默认spawn的环境X/Y偏移（米）|物体reset偏航|高处转向最大指令增量|
|---|---|---:|---:|
|home-fast|[0,0]|0度|0.5度/50 ms|
|home-slow|[0,0]|0度|0.25度/50 ms|
|left-yaw20|[-0.02,0.02]|+20度|0.25度/50 ms|
|right-yawm20|[0.02,-0.02]|-20度|0.25度/50 ms|

每种条件分别seed=0、1，各重新创建一次环境，共2组增益×4条件×2次=16次抓取。
位置和朝向范围是固定值，不是连续随机抽样；改变seed也不等于得到了独立随机物体场景。
这轮用于检查局部几何适应和可重复性，不能将通过次数直接解释为通用抓取成功率。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/grasp_robustness_cases.json \
  --grasp-scenarios-json openpi_isaac_bridge/grasp_robustness_scenarios.json \
  --modes grasp \
  --record-physics
```

加入抓取场景选项时只允许运行grasp模式，不能同时指定stream场景。
场景名、base_case、seed、物体偏移、reset偏航及转向步长记录在汇总和启动命令中；每次运行单独输出。
实际物体初始位姿、抓取目标四元数及测得的物体偏航也在 `grasp_result` 中保留。

抓取朝向现在按物体在根坐标中的实际初始偏航对齐：`Rz(object_yaw) * Rx(pi)`，四元数使用xyzw。
物体仍为平放方块，只处理偏航，不支持任意倾斜物体的抓取姿态推断。
高处orient阶段可将最大旋转增量减半；接近、下降、闭爪、抬升、保持阶段沿用原控制与阈值。
转向最大步数按限速同比延长，避免慢转向仅因更慢而被原超时误判。
指令增量限制不是实际角速度保证；比较时应同时看orient阶段耗时、估算力矩峰值与触及上限样本数。

`[Grasp] success=True`仍要求物体持续抬升至少35 mm并保持2秒，相对位姿稳定；
空闭、滑落、过大物体运动或阶段未到位的记录保留为失败，不能靠等待固定时间认定成功。
当前代码仍使用仿真物体真值和任务原始重力配置，不代表视觉定位、OpenPI闭环或真实机器人验证。

本轮新增配置、偏航适配和路由通过CPU检查；实际偏航物体接触与慢转向力矩还需Spark实测。
运行时间会明显长于上一轮三次测试：共16次独立启动，每次有完整接近、闭爪和抬升流程。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add repeated grasp pose and orientation-rate results"
git push origin main
```

## 首轮OpenPI模型驱动的小步闭环

确定性控制与夹持已经完成局部验证；本轮开始执行模型给出的末端动作。
使用单一已测候选：关节2/4为5400/120，其它五个机械臂关节900/20，原力矩/速度限制不变。
这是输入/动作接口与小步响应测试，夹爪始终打开，无抓取成功判定。

如果OpenPI服务尚未运行，先在Spark终端1启动并保持运行：

```bash
cd /home/test/Robot/openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

终端2运行（之前已安装IsaacLab项目的openpi extra时直接执行）：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_probe_cases.json \
  --modes openpi-step \
  --server-host 127.0.0.1 --server-port 8000 \
  --openpi-steps 20 --openpi-state-frame world \
  --prompt "pick up the cube" \
  --record-physics
```

运行器在该模式自动添加 `--extra openpi` 和物理采样。若项目尚无该extra，按文首准备章节安装客户端。
不添加--query：--query始终是只保存动作的旧入口；openpi-step是显式执行入口。
本轮固定物体reset位置与seed=0，初始腕部朝向由固定场景点[0.5,0,0.04]一次性设置，循环中不重新瞄准物体。
模型观测不含物体真值；物体真值仅用于额外停止检查。

每次循环：
1. 读取当前两路RGB及末端/夹爪状态，保存模型实际收到的224x224 RGB PNG。
2. 查询服务，保存完整返回动作块并检查所有元素有限、形状为N×7。
3. 只取第一个动作，按robosuite 1.4默认OSC缩放解码，再分别将平移/旋转向量范数限制到1 mm、0.25度。
4. 将世界轴增量转换为Isaac机器人根轴目标，通过当前IK执行一次50 ms仿真控制步；夹爪请求记录但强制应用打开。
5. 保存实际末端变化和两指状态，再重新采图并查询，最多20次。后续chunk动作不缓存执行。

最多累计20 mm名义平移和5度名义旋转；执行保护另限制距起点30 mm、姿态变化6度，
以及实际单周期位移10 mm、转角3度和关节软限位。物体移动超过20 mm或状态无效也会停止。
动作显著限幅会改变原模型策略行为，本轮不能据没有完成抓取评判模型能力。

**模型输入约定与尚未对齐的部分**

OpenPI官方LIBERO示例使用末端世界坐标位置、旋转向量与两指qpos，见
[examples/libero/main.py](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。
本轮默认state-frame=world，额外记录root状态便于对比；--openpi-state-frame root只用于显式对照，不能替代场景配准。
当前输入未自动添加LIBERO桌面高度或其它坐标平移，世界坐标原点、TCP定义和相机分布仍需检查。

两个已确认的接口差异在本轮适配：
- 旋转向量按官方_quat2axisangle使用xyzw四元数计算，不使用Isaac默认的w>=0最短转角规范化。
  两者在接近/超过180度时可产生不同state数值，虽然对应物理朝向等价。原始四元数同时保留。
- 当前Isaac Panda两指物理位置均为正；对应robosuite 1.4 Panda第二指qpos为负，
  输入映射为[q1,-q2]，见[官方Panda XML](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/assets/grippers/panda_gripper.xml)。
  检查Isaac指关节顺序后才转换，原始两指读数单独保存。这不代表checkpoint归一化统计已经逐项验证。

图像用RGB resize/pad到224，不自动照搬MuJoCo示例的180度翻转；实际模型输入图片随结果导出供检查。
当前仍假定LIBERO与Isaac世界轴方向一致；没有证明场景坐标、TCP、相机外参与训练分布已经完全匹配。

**运行与结果判读**

控制步长50 ms，但等待推理时同步暂停仿真。20次成功执行只推进1秒仿真时间，墙钟耗时包含推理、渲染和启动。
这不是实时20 Hz控制，不是官方默认动作块执行策略，也未证明IK与OSC动力学等价。

`[OpenPI step] 1/20...`记录请求耗时、模型夹爪请求以及实际保持打开。
`openpi_result.probe_completed=True`只表示所有受限步骤执行完成，不能解释为抓取成功或模型对齐。
`completed_steps`记录已完成的控制步；发生执行后保护异常时，execution_attempted与executed字段区分动作是否已经尝试。
连接/推理抛错、动作非法或保护停止会记录failure_reason及现有trace，仍可作为已完成的诊断导出。
若客户端仍在等待连接，先确认终端1服务已正常启动。

汇总保留：完整动作块、裁剪/范数缩放、请求和实际夹爪指令、输入state、两参考系状态、原始四元数、
目标与实际末端变化、推理墙钟耗时、实际增益及物理步关节记录。
`observation.json`仍是启动时的旧采集记录；模型实际每次收到的输入以 `openpi_result.trace.state_sent` 和对应PNG为准。
输入图片全部保存于运行目录；通用导出器自动复制第一/最后一次输入的两张PNG到diagnostics/gain_sweep，方便从GitHub复核。

新路径通过CPU流程/限幅/状态转换/导出检查，真实服务调用和Isaac运行仍待Spark验证。
完成后推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add first bounded OpenPI model-motion probe results"
git push origin main
```


## 固定 OpenPI 动作分量重放（2026-10-29）

上一轮 `239e5656347f0c5a40ca11424e92a3e098bc5d54` 的20条实际首动作已冻结在
`openpi_replay_actions.json`，不是重新向模型查询。使用原来的1mm/0.25度范数限幅，
再分别删除旋转或平移分量；混合组保留两者。保留分量在三组间完全一致。

每组独立启动新环境、同一seed和物体位置、同一关节增益，先核对记录起始姿态
（位置误差≤0.25mm、角度≤0.1度）；不符则停止并保存诊断。每组执行20条50ms动作，
再执行20条50ms零增量，夹爪始终打开。零增量的目标仍由当前反馈姿态产生，
不是固定末端目标保持，也不是期望姿态累加。无需启动OpenPI服务或加载模型。
保留30mm/6度总运动范围、单周期10mm/3度、关节软限位与物体移动检查。

在Spark仓库运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_probe_cases.json \
  --modes openpi-replay-translation openpi-replay-rotation openpi-replay-mixed \
  --record-physics
```

三组各40个控制步、200条10ms物理记录，合计应为600条。Viser仍在8080端口。
`replay_result.trace`记录目标/实际位姿、各周期位置与角度误差、投影增益、原始动作与限幅动作；
`period_error_summary`汇总前20个运动周期的RMS和最大误差。
`replay_setup`包含源提交和动作文件SHA256。检查`probe_completed`，
外层`status=completed`仅表示诊断记录已生成，不表示跟踪误差通过阈值。
本轮不自动筛选增益，也不预设新的通过阈值；先比较三类运动的误差和10ms关节响应。

若纯分量误差较小而混合组明显增大，说明存在组合运动耦合或动态滞后，需要继续定位；
若三组都有大误差，则继续检查共享控制链路。仅凭这次重放不能证明模型输入、
场景坐标/TCP已对齐LIBERO，也不能证明抓取能力。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add frozen OpenPI component replay results"
git push origin main
```

本地CPU单元测试覆盖分量保留、零输入尾段、起始姿态不符停止、故障记录与三种模式路由；
Isaac/PhysX实际执行仍需在Spark运行核验。


## 旋转位置偏移：单关节单参数对照（2026-10-29）

固定分量重放结果86a83228f0bc显示：纯旋转的周期位置偏移RMS为0.272mm、
最大0.506mm；混合组RMS为0.375mm、最大0.803mm。混合组复现原始模型轨迹，
其每周期位移近似为纯平移与纯旋转响应之和。尚不能从关节误差大小直接确定TCP误差贡献。

本轮用 `replay_joint_gain_cases.json` 做9组增益配置：1组原基准，加上J1/J3/J6/J7各2组。
每次只改变一个关节的一个参数：

| 对照 | 被测关节Kp | 被测关节Kd |
|---|---:|---:|
| 原基准 | 900 | 20 |
| 刚度增加25% | 1125 | 20 |
| 阻尼增加25% | 900 | 25 |

J2/J4始终为5400/120，未被测关节始终900/20。每种配置都执行纯平移、纯旋转、混合重放，
共27个独立仿真。动作、限幅、50ms周期、物体位置、夹爪打开及原有检查保持一致。
这是一组有限差分敏感性实验，不预设增加刚度或阻尼一定有效，不更改力矩上限，也不自动部署最佳配置。
若提高阻尼减少旋转位置偏移却增大平移跟踪误差，需要考虑两者权衡，不能只看单项改善。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/replay_joint_gain_cases.json \
  --modes openpi-replay-translation openpi-replay-rotation openpi-replay-mixed \
  --record-physics
```

无需OpenPI服务。每组40控制步，200条10ms物理采样；全部完成应得到27条run记录、5400条物理采样。
先检查每条 `replay_result.probe_completed=true` 和 `completed_steps=40`，再比较：

1. 纯旋转及混合组的周期位置RMS/最大值，以及角度RMS/最大值。
2. 纯平移是否退化，逐周期投影增益是否改善。
3. 七个关节10ms目标/响应、力矩估计峰值与限值余量。
4. 后20步零输入的净漂移（不是固定目标稳态误差）。

该轮同时修复起始姿态角度检查：先归一化四元数并考虑正负等价，避免相同float32四元数
因范数舍入产生约0.04度伪误差。检查阈值仍为0.25mm/0.1度，动作执行与限幅计算不变。
场景/TCP与LIBERO的注册仍不在本轮验证范围。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add single-joint OpenPI replay gain sensitivity results"
git push origin main
```


## Y向误差定位：实际IK雅可比与关节贡献（2026-10-29）

上一轮a2b47d546df8的27次测试没有得到全面优于基准的配置。基准纯旋转位置误差
约99.6%的平方均值来自Y分量；J1加刚度明显放大误差，J3加刚度改善部分峰值，
但不能据此把关节角误差直接认定为各关节造成的TCP误差。

本轮回到 `openpi_probe_cases.json` 中的原基准，仍重放同一份动作，使用
`--record-jacobians` 读取每个实际10ms物理子步的运动学数据，不更改增益或动作。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_probe_cases.json \
  --modes openpi-replay-translation openpi-replay-rotation openpi-replay-mixed \
  --record-jacobians
```

无需启动OpenPI服务。该开关自动开启物理采样，仅允许冻结重放。三组各40个控制步，
共600条物理记录，应与既有基准的轨迹和周期误差复现一致。记录会增加墙钟耗时，
但不增加模拟步、不延长模拟控制周期。

**采样方式与坐标约定**

- 包装实际动作项 `_compute_frame_jacobian` 调用：原函数只调用一次，原结果原样返回，
  只保存detach/clone副本；不重新调用雅可比函数、不修改驱动目标。
- 同时记录IK点、传感器TCP的root位姿、IK期望位姿、七个关节目标和实际值。
  明确两点是同一手部刚体上的固定偏置，固定基座、关节顺序与6×7矩阵尺寸须通过检查。
- 实际IK雅可比预期表示root轴中的几何速度。传感器点用
  `Jv_sensor = Jv_ik + Jw × (p_sensor - p_ik)`逐列转换，角速度行不随固定点平移改变。
- 将本机IK方法和jacobian_w/jacobian_b属性源码保存为 `jacobian_implementation.txt`，
  以便核对安装版本。若接口缺失、采样时序不匹配或出现非有限值，会停止，不伪造雅可比。

接口参考：[Isaac Lab动作项源码](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py)。
实际数据以本机保存的源码及位移预测残差为准，不假定安装版本与main完全相同。

**每条物理记录增加的内容**

`kinematics_before`包括实际使用的IK雅可比、积分前关节角、两点位姿和IK期望位姿；
`kinematics_after`是积分后的两点位姿；`sensor_jacobian_root`是转换后的传感器点矩阵。
`kinematic_attribution`包含：

1. 逐关节一阶跟踪贡献 `J_sensor[:,j] * (q_target[j] - q_after[j])`，保留符号，
   以及七关节贡献之和，不能把各关节RMS直接相加或当作独立贡献百分比。
2. 请求的传感器目标与IK期望位姿所隐含的传感器目标之差，检查控制点目标转换。
3. IK目标关节增量的一阶残差。
4. 真实TCP位移减 `J_sensor * (q_after - q_before)`，检验该局部雅可比预测的可信度。
5. 真实传感器目标误差及代数记账闭合误差。

记账公式为：
`真实目标误差 = 控制点目标映射误差 + IK一阶残差 + 关节跟踪贡献之和 - 实际位移一阶残差`。
闭合是代数恒等检查，不是独立验证，更不是动力学因果证明。
必须先看真实位移与雅可比预测是否接近，再解读J1/J3等关节Y向贡献是否互相抵消、是否时序失配。

汇总保存在 `physics_joint_summary.json` 的 `kinematic_attribution_summary`，并进入导出的summary。
它按动作段/零输入尾段汇总全部10ms样本，和前20个50ms周期末的RMS不是同一采样集合。
原始压缩trace包含全部矩阵与带符号向量，可单独选每周期第5个子步比较。
安装代码、矩阵语义、位移残差或非线性效应若不能支持一阶解释，应先排查这些因素，再调整增益。
本轮仍不验证LIBERO场景注册或抓取成功。

运行完成后：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add baseline TCP Jacobian attribution replay results"
git push origin main
```

本地CPU验证覆盖刚体点变换符号、相反关节贡献、目标点旋转映射、时间对齐、
只调用一次原雅可比及恢复方法、既有物理步数不变、开关路由和源码导出。
实际Isaac接口与仿真轨迹须由Spark运行验证。


### 雅可比初始化刚体身份检查修复

若在动作开始前出现 `Sensor and IK must be rigid offsets on the same resolved hand body`，
旧检查比较的是传感器路径末段字符串与动作项私有 `_body_name` 字段，不能可靠表示解析后的刚体身份。
单凭旧错误信息无法判断是名称/正则/容器表示差异，还是真实刚体不同。

修复后分别调用本机 `robot.find_bodies` 解析传感器路径末段和IK配置模式，再检查：
两者均唯一匹配，解析后的刚体索引相同，且与动作项实际使用的 `_body_idx` 相同。
不再把私有 `_body_name` 的字符串或列表表示作为等价性依据。真实不同的刚体或歧义匹配仍然停止，
不会为了继续测试而假定两个不同刚体是刚性连接。

运行器会显示 `[Jacobian bodies]`，包含实际路径、配置模式、名称、索引和私有字段repr；
同样内容保存为该次输出目录的 `jacobian_body_resolution.json`，成功时也进入物理汇总。
若仍报错，错误消息会列出解析结果，可据此修正配置或另行验证刚体连接，而不是绕过检查。

更新后重新执行上节同一条三组重放命令。失败的那轮不应导出为新结果；
成功完成后再导出最新汇总并推送。不会修改模型、增益、动作或仿真周期。


### 完整USD路径正则修复

截图中的 `re.error: unbalanced parenthesis at position 0` 暴露出路径不能按 `/` 拆成
“末段正则”：完整表达式的分组或后缀可能跨越路径分隔符，例如
`({ENV_REGEX_NS}/Robot/panda_hand)(|/)` 的最后一段只是 `)`。
上一版把这种片段交给 `find_bodies` 的处理方式已移除。

当前实现先从活动USD stage读取具有RigidBodyAPI的实际prim路径，展开命名空间占位符，
对完整传感器路径正则做fullmatch，再检查匹配的实际刚体属于机器人路径。
仅对匹配后的字面USD路径取刚体名称，按 `robot.body_names` 映射到索引，
最后核对IK实际使用的body索引。完整表达式、匹配到的USD路径、刚体清单和任何解析错误
都会写入 `jacobian_body_resolution.json` 并输出 `[Jacobian bodies]`。
多重匹配、错误机器人、缺失刚体和实际索引不一致仍会停止。

分组跨越 `/`、路径内部备选分组、路径后缀以及歧义/错误归属已加入CPU回归测试。
这些测试不替代Spark上真实USD和Isaac接口验证。更新后仍运行同一条三组重放命令；
失败轮不导出，成功完成后再导出最新结果。


## J1/J3成对增益对照：保持TCP定义和动作不变

雅可比采样结果e5d64f08784d显示：基准50ms周期末的Y向目标误差主要来自J1/J3
跟踪残差。纯旋转第9步末，两关节一阶贡献约−0.149mm、−0.345mm，实际误差约−0.506mm；
目标点映射项约+0.010mm。全过程和周期末的贡献符号关系不同，不能仅靠一个关节的RMS判断。

本轮配置文件为 `replay_j13_pair_cases.json`，共9种：

|配置|J1 Kp/Kd|J3 Kp/Kd|
|---|---|---|
|j13-pair-baseline|900/20|900/20|
|j13-k1125-900-d20|1125/20|900/20|
|j13-k900-1125-d20|900/20|1125/20|
|j13-k1125-1125-d20|1125/20|1125/20|
|j13-k900-900-d25|900/25|900/25|
|j13-k1125-900-d25|1125/25|900/25|
|j13-k900-1125-d25|900/25|1125/25|
|j13-k1125-1125-d25|1125/25|1125/25|
|j13-cross-j1-d25-j3-k1125|900/25|1125/20|

前8组形成J1刚度×J3刚度×共同阻尼的小范围组合，含原基准和此前单独加刚度的对照。
第9组检查此前J1增加阻尼、J3增加刚度组合后的效果，不能用两次单关节改善量直接相加预测它。
这些都是候选实验，不预设更高参数一定更好，不自动选择或部署新默认值。
J2/J4保持5400/120；J5/J6/J7保持900/20；力矩限值、TCP偏置、IK设置、冻结动作、
1mm/0.25度限幅、50ms控制周期和夹爪打开设置保持原样。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/replay_j13_pair_cases.json \
  --modes openpi-replay-translation openpi-replay-rotation openpi-replay-mixed \
  --record-jacobians
```

无需OpenPI服务。每种配置单独启动三组环境，使用相同起始状态和固定动作。
每次40个控制步、200条10ms记录；全部完成应为27次测试、5400条物理采样。
雅可比捕获和完整USD路径解析沿用上一轮已实际跑通的实现，本轮只增加配置和说明。

判读顺序：

1. 确认每次 `probe_completed=true`、`completed_steps=40`，并核对实际执行器增益。
2. 先比较20个50ms周期末的Y向位置RMS/峰值，同时看总位置和角度RMS/峰值。
3. 纯平移作为回归检查，不能用旋转改善换取明显的平移退化而不说明权衡。
4. 对比J1/J3带符号贡献，检查第9/10步及其他换向处是否减少同向叠加。
5. 检查全部物理子步的力矩估计、零输入尾段漂移，以及雅可比位移预测残差。

代数闭合不是独立准确性验证；不能把关节贡献RMS相加成误差百分比。
全部10ms样本的RMS与每周期第5个子步的RMS必须分别统计。
本轮不调整目标点映射，以免同时引入另一类变量；映射问题可在后续独立处理。
若没有综合改善，保留原基准并重新评估控制策略，不因测试完成就认定找到了最佳增益。

运行完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add paired J1 J3 gain replay and attribution results"
git push origin main
```

本地已用现有配置验证器检查9种配置，并通过真实增益配置函数验证各关节目标值及限值保留。
实际动态效果仍须由Spark测试确认。


## J1/J3候选多位置、多幅度回归

成对调参结果8ced3944aefa中，J1/J3均1125/25在原始姿态、固定动作的12项
动作段位姿指标上均优于原基准。该结果需要用其他关节构型和运动方向验证。
本轮复用已运行过的六轴确定性往返流程，只新增增益配置文件，不修改仿真、采样或控制代码。

使用 `j13_workspace_regression_cases.json` 比较：

|配置|J1/J3|J2/J4|J5/J6/J7|
|---|---|---|---|
|workspace-j13-baseline|900/20|5400/120|900/20|
|workspace-j13-k1125-d25|1125/25|5400/120|900/20|

仍使用原来的力矩限值、TCP定义与IK配置。通过既有
`cartesian_workspace_scenarios.json` 展开三种起始位置、两种幅度：

|起始位置|相对初始TCP的root坐标偏移/mm|
|---|---|
|home|[0,0,0]|
|yplus|[0,+15,+10]|
|yminus|[0,−15,+10]|

末端保持初始朝向。这是三个不同起始位置对应的关节构型，不包含额外起始朝向变化。
每个位置测试X/Y/Z平移与RX/RY/RZ旋转，各自执行正向、停止、反向、停止。
1倍幅度每步平移0.25mm或旋转0.1度，20步名义累计5mm或2度；
2倍幅度每步0.5mm或0.2度，20步累计10mm或4度。不是每步移动5/10mm。
每个50ms周期仍基于当前测量位姿生成增量目标，不累计期望位姿、不等待单条指令收敛。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/j13_workspace_regression_cases.json \
  --stream-scenarios-json openpi_isaac_bridge/cartesian_workspace_scenarios.json \
  --modes libero-stream \
  --record-physics
```

无需启动OpenPI服务。此处使用 `--record-physics` 记录七个关节的目标/响应，
`--record-jacobians` 当前只支持冻结动作重放，不能加入这条命令。
本轮六轴分别往返用于控制器回归，不是多位置混合OpenPI动作重放或抓取测试。

正常完成应为2套增益×6个位置/幅度条件=12次独立测试，共72条轴结果。
每次80个预定位控制步，要求最后5步位置误差≤0.25mm、角度≤0.1度，然后运行6轴×70步。
每次500个控制步、2500条10ms物理记录，合计30000条；初始化warmup不计入这些记录。
预定位不达标会停止该轮，不能把未到位结果当作合格回归。

判读时同时检查：

1. `cartesian_stream_screen.all_passed`：完整六轴、累计正向投影增益0.9–1.1，且原有检查全部通过。
2. 原检查包括回程位置≤1mm/角度≤0.3度、单周期位置误差≤1mm/角度≤0.5度、停止漂移≤0.5mm/0.2度及夹爪打开。
3. 按相同位置与幅度成对比较两套增益的周期误差RMS/峰值、停止漂移和力矩估计余量。
4. 单独查看Y、RX/RY/RZ及高幅度条件，避免仅依靠一个总通过数忽略性能退化。

累计投影增益的通过不代表每条指令50ms内完全到位。边界值轻微超限需要看原始数据，
不自动等同于明显失稳。通过全部小范围回归也不代表全工作空间或实际抓取均已验证。
保持基准和候选两套参数，验证后再决定是否进入下一步模型闭环，不自动改默认配置。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add J1 J3 candidate workspace regression results"
git push origin main
```

配置已通过现有验证器、场景展开和实际增益配置函数的校验，确认12种组合及其他关节和限值保留。
真实多条件动态效果待Spark运行结果确认。


## J1/J3候选确定性抓取回归

多位置六轴回归16add29445b9中，原基准和J1/J3=1125/25均通过检查；候选改善Y向
周期位置误差和停止漂移，但RX角度RMS存在小幅退化。下一步验证接触、夹持和抬升时的表现。
本轮复用原有抓取宏和判据，只新增两套增益配置及四个固定场景。

配置：`j13_grasp_regression_cases.json`。
原基准J1/J3=900/20，候选J1/J3=1125/25；两者J2/J4=5400/120、J5/J6/J7=900/20。
力矩限值、物体资产/质量/摩擦、TCP和IK设置沿用原流程，不自动更改默认增益。

场景：`j13_grasp_regression_scenarios.json`。

|场景|方块XY偏移/mm|方块yaw|朝向阶段每步旋转上限|
|---|---|---:|---:|
|home-fast|[0,0]|0度|0.5度|
|home-slow|[0,0]|0度|0.25度|
|left-yaw20|[−20,+20]|+20度|0.25度|
|right-yawm20|[+20,−20]|−20度|0.25度|

快慢仅改变原位朝向调整阶段的旋转增量上限，其余阶段设置相同。四个场景均seed=0，
每次独立创建环境，属于固定条件回归，不把不同seed下重复同一确定性结果当作统计成功率。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/j13_grasp_regression_cases.json \
  --grasp-scenarios-json openpi_isaac_bridge/j13_grasp_regression_scenarios.json \
  --modes grasp \
  --record-physics
```

无需OpenPI服务。该流程使用仿真物体真实位姿和yaw生成自上而下抓取目标，属于
确定性控制器/接触回归，不是视觉模型自主抓取。机器人重力设置沿用原任务并写入grasp_setup。
不要加入仅支持冻结重放的 `--record-jacobians`。

流程为：打开夹爪→原位调整朝向→移动到物体上方100mm→下降至物体中心上方5mm的TCP目标
→闭合80步→检查夹持候选→抬升50mm→保持40步（2秒）。
各移动阶段都需要连续5步位置误差≤2mm、角度误差≤2度才能进入下一阶段。
原有夹持检查包括双指稳定、阻挡候选和物体靠近TCP；它本身不等于抓取成功。
最终成功还要求：

- 保持阶段物体最低抬升高度≥35mm。
- 保持阶段物体相对TCP位移≤10mm、相对旋转≤10度。
- 从闭合到抬升/保持的相对位移变化≤15mm，手指保持非空闭合。
- 保持阶段末端位置误差≤2mm、角度误差≤2度。

原有掉落、过快物体运动、抬升前物体偏移及移动超时检查继续有效。
记录是仿真状态和控制周期采样，夹持判据不是接触力测量，也不代表真实机器人负载验证。

正常生成8条运行记录，各次阶段收敛步数会不同，因此不预设固定物理采样总数。
每次应有 `physics recorded_substeps = 5 × grasp_result.recorded_controls`。
外层 `status=completed` 仅代表诊断输出已生成，必须逐条检查 `grasp_result.success`
及失败原因、阶段步数、抬升高度、相对滑移、末端误差和10ms力矩估计峰值。
相同场景下成对比较两套增益，尤其RX大角度朝向调整和带载抬升；不能只看总成功数。

完成后导出并推送（抓取判据失败但诊断正常完成时，失败记录同样需要保留分析）：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add J1 J3 candidate deterministic grasp regression results"
git push origin main
```

两套配置和四个场景已通过既有验证器、组合展开及增益实际应用检查。
运行器、抓取宏与成功阈值沿用已有实现，实际效果由Spark结果确认。


## 候选增益接回OpenPI：20步受限闭环

确定性抓取结果2e1ab34b8e42中，原基准与J1/J3=1125/25在四个固定场景均通过。
这说明候选可继续用于模型接口测试，不代表OpenPI自身已经具备当前场景的抓取能力。

本轮使用 `openpi_j13_candidate_cases.json`：J1/J3=1125/25，J2/J4=5400/120，
J5/J6/J7=900/20，其余限值和TCP定义保持原样。
重新读取实时图像和状态并调用模型，不是冻结动作重放或使用物体真实位姿规划抓取。

终端1：若已有LIBERO模型服务在8000端口运行，可继续使用；否则沿用之前已跑通的启动方式：

```bash
cd /home/test/Robot/openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

等待服务开始监听，再在终端2运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_j13_candidate_cases.json \
  --modes openpi-step \
  --server-host 127.0.0.1 --server-port 8000 \
  --openpi-steps 20 --openpi-state-frame world \
  --prompt "pick up the cube" \
  --record-physics
```

此命令通过现有openpi extra加载客户端；如果此前尚未安装，可在IsaacLab项目中执行
`uv add --optional openpi --editable /home/test/Robot/openpi/packages/openpi-client`，本轮已安装过则无需重复。
Viser仍为8080端口。无需加入仅支持冻结重放的record-jacobians开关。

**执行语义**

每次使用当前两路RGB图像和8维状态发起一次新推理，接收完整N×7动作块并验证有限值，
只执行第一条动作；随后仿真推进50ms，再重新观察。共20次推理、20个控制步，
正常完成应得到1条run记录、100条10ms物理样本。
每步平移范数上限1mm，旋转向量范数上限0.25度，夹爪始终打开；模型的开合请求、
原始动作块、控制器域解码结果、限幅比例和实际执行动作均保留。
原有30mm/6度总运动范围、单周期变化、关节软限位、物体移动等停止检查保持有效。

图像仍为RGB resize/pad到224，不做180度翻转；状态仍采用world位姿、LIBERO示例的
带符号旋转向量和双指[q1,−q2]。这些约定沿用之前实现，场景/TCP与训练视角注册仍待独立验证。
物体真值仅用于诊断和停止检查，不进入模型输入，也不用于动作目标生成。

同步推理会暂停仿真，本轮不验证真实墙钟20Hz。只执行动作块第一条并施加强限幅，
也不同于模型原始完整动作块的自由执行。最多20mm名义路径不足以保证接近远处物体，
本轮用于检查闭环方向和输入/输出协议，不评价抓取成功。

**增加的服务信息记录**

首次连接后使用官方客户端的缓存握手接口 `get_server_metadata()`，保存到
`openpi_setup.server_metadata`。接口参考：
[OpenPI客户端源码](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py)。
记录明确区分reported、empty、unsupported、unavailable等状态；元数据为空不等于没有模型，
也不证明实际检查点或归一化统计正确。需要结合服务端启动日志确认实际加载的配置、权重和统计。
元数据只作为数据记录，不作为改变动作或执行指令的来源；读取失败不会改变既有动作逻辑。

核对以下项目：

1. `openpi_result.probe_completed=true`、`completed_steps=20`；外层completed仅表示诊断文件生成。
2. 当前输入图像、状态、返回动作块维度与有限性，实际是否仍每次只执行第一条。
3. 原始平移/旋转被限幅的比例、模型夹爪请求与实际打开指令的区别。
4. 末端真实位移、周期位置/角度误差、方向、相对物体的诊断距离和物理步力矩估计。
5. 服务端实际加载的模型信息。服务元数据不含检查点时，不凭--env或文件名断言身份已验证。

模型推理可能具有随机性；不能把本轮新输出和以前不同的一组动作直接当成只改变增益的严格对照。
也不能仅凭远处出现关闭请求判定策略一定错误，尤其在动作被强烈压缩、动作块只执行首条的条件下。
下一阶段是否放开夹爪或扩大执行范围，应依据本轮轨迹、输入与动作约定核验结果决定。

完成后导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add candidate-gain bounded OpenPI closed-loop results"
git push origin main
```

本轮候选配置已通过验证器检查；增加的握手记录和既有流程共99项CPU单元测试通过。
实际模型连接和Isaac仿真效果待Spark运行验证。


## OpenPI动作块执行长度1与5对照

结果72d9e5b9572f已验证候选增益下20步首动作闭环完成，但所有平移均受限，夹爪覆盖打开，
TCP仍远离物体，握手信息为空。本轮比较重规划长度，并保存服务端实际启动日志作核对。

两组均使用 `openpi_j13_candidate_cases.json`，相同增益、场景初始化、20个50ms控制步、
每步1mm/0.25度范数限幅、夹爪始终打开及原有运动边界：

|条件|每次推理执行动作块前缀|总控制步|正常推理请求数|物理采样|
|---|---:|---:|---:|---:|
|chunk1|第0条|20|20|100|
|chunk5|第0–4条，依次执行|20|4|100|

官方LIBERO示例默认replan_steps=5：
[官方执行循环](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。
这里只对照块长，仍保留限幅和开爪，因此不等于完整复现LIBERO控制或抓取。

终端1：在原服务终端Ctrl+C停止旧服务，再重新启动并保存实际启动日志；确认8000端口开始监听。
不要让旧服务占用8000而把新服务启动失败的日志误当作有效加载证据。

```bash
cd /home/test/Robot/openpi
PYTHONUNBUFFERED=1 \
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000 \
  2>&1 | tee /home/test/Robot/openpi/openpi_chunk_probe_server.log
```

终端2：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_j13_candidate_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 1 5 \
  --openpi-steps 20 --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" \
  --record-physics
```

运行器在同一汇总中生成chunk1和chunk5两组唯一名称，分别新建环境。正常完成是2次run、
40个控制步、200条物理采样；模型调用分别20次和4次。不要只运行两条独立命令后导出latest，
否则latest只会指向最后一轮，不能自动保留两组对照。

**调度与记录**

- 每次推理验证整个返回块为有限N×7，且长度不小于所需前缀；不足则在执行前停止。
- 同一块的前5条只各执行一次，按0、1、2、3、4排序，不重复第0条。
- 每个控制步仍读取新图像和状态用于诊断；仅在需要新推理时发送给模型。
- trace中的 `inference_performed` / `observation_used_for_inference` 表示是否实际调用模型；
  `chunk_id`、`chunk_action_index`、`chunk_observation_step`记录来源，`selected_raw_action`记录本步选择。
- `raw_action_chunk`在同一块内重复保存作追溯，不代表又调用了一次模型。
- `state_observed`为当前观测；缓存动作步的 `state_sent=null`、`inference_wall_s=null`，不能误当新输入或零耗时推理。
- 缓存期间的openpi_input图片名沿用旧格式，但只是诊断图片；是否真实输入须看标志。
  preview_images仍导出第1和第20控制步之前的两路观测，chunk5末张通常不是实际推理输入。
- openpi_result增加 `replan_steps`、`inference_requests`、`validated_inferences`。

每个动作仍由当前测量位姿生成相对目标，不累积期望位姿、不额外等待收敛。
只在两组都正确执行20步后比较末端轨迹、到物体距离变化、限幅比例、夹爪请求和跟踪残差。
若途中停止，保留部分块、失败动作索引和记录，不能把不完整轨迹直接当成完整组比较。
夹爪请求仍仅记录，不执行；物体真值只用于诊断/停止检查，不生成模型动作。

服务日志由 `--policy-server-log` 显式指定，运行完成时读取前128KiB，保存在每组输出目录的
`policy_server_startup.log`。摘要记录来源路径、捕获时间、截断标志和前缀SHA256，导出时随诊断结果复制。
日志应核对实际配置、权重和归一化统计路径；若未打印相关信息，身份仍未核验。
它是用户提供的日志证据，不独立证明该进程就是端口服务所有者；握手元数据也继续保留。
日志内容只作数据记录，不作为执行指令或修改模型参数的依据。

两组先后使用同一模型服务。服务端随机数/内部状态可能随调用推进，且输入轨迹会分叉；
一次对照属于执行方式探索，不能把所有差异严格归因于块长，也不是增益对照或统计成功率。
若要作严格因果比较，后续需固定推理随机条件或重复试验。
推理期间仿真同步暂停，不据此宣称真实墙钟20Hz；本轮总路径仍很短，不评估完整抓取。

完成后：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add bounded OpenPI chunk1 versus chunk5 results"
git push origin main
```

108项CPU测试通过，包括动作顺序、仅在块边界推理、末块截断不增步、非法块拒绝、
中途停止、运行器两组路由及服务日志大小限制/导出。实际接口与模型效果待Spark验证。


## 块长5：最多60步分段接近测试

块长对照aba4bb07991e中，块长5已正确执行，原始动作显著受限，TCP仍距物体约351mm。
本轮以块长5继续检查更长的接近趋势，保持候选增益、1mm/0.25度每步范数限幅和夹爪打开。
本轮不评价抓取，也不自动开启夹爪执行模型请求。

运行器新增显式 `--openpi-approach`，只有openpi-step且块长5可用；不启用时仍最多20步。
启用后最多60个50ms控制步，20步为一段，总计最多12次推理、300条10ms物理记录。
模型仍每5步重新观察并推理，物体真值只用于监测与停止，不生成动作目标。

继续使用上一轮8000端口模型服务和启动日志。若服务已停止，可在终端1按上节命令重启，
确认开始监听并保留 `openpi_chunk_probe_server.log`；无需为本轮强制重启仍正常运行的服务。

终端2运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_j13_candidate_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 5 --openpi-steps 60 \
  --openpi-approach \
  --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" \
  --record-physics
```

输出配置名称增加 `-approach`，可区别此前20步chunk5结果。

**分段监测与停止条件**

- 每20步记录本段起止TCP到物体中心的世界坐标欧氏距离、减少量及净TCP位移。
- 单段距离减少不足2mm，则在该段末停止，不继续后续模型动作。
- 任一控制步末，距离比整轮起始距离增加超过5mm，则停止。
- 距离进入80mm边界时停止；若初始化已在该范围内，推理和动作均不开始。
- 原有物体移动超过2cm、非有限状态、关节软限位、单周期TCP变化等检查继续有效。

分段阈值是本轮诊断条件，不是训练奖励、接近目标生成器、碰撞保证或抓取成功判据。
距离检查在控制步执行后进行，不能阻止该步已经发生的运动。
为容纳60步，只有本模式的相对起始TCP总位移边界由30mm改为70mm，净旋转边界由6度改为18度；
每步1mm/0.25度限幅和原有单周期10mm/3度异常变化检查不变。普通20步和冻结重放模式边界不变。
60步的名义平移路径最多60mm，仍可能远离方块；本轮只检查接近趋势，不保证到达物体。

控制台 `[OpenPI segment]` 在第20/40/60步输出距离和本段进展。
结果 `openpi_result.approach_summary` 保存阈值、分段记录、初始/最后距离及停止原因。
正常执行完60步时 `probe_completed=true`、completed_steps=60；若监测停止，则为false，
并有 `stopped_by_monitor=true`、`termination_reason`：

- `segment_progress_less_than_2mm`
- `distance_increased_more_than_5mm`
- `near_object_boundary`

这些是受控停止，不等于程序崩溃或模型必然失败；已执行步仍记入completed_steps。
外层completed只代表诊断记录已生成。若停在20或40步，也应导出分析，不为凑满60步关闭监测。
物理采样数量应等于已执行控制步的5倍（其他异常失败需结合trace和错误信息核对）。
预览图导出第1步、各20步检查点附近及最后记录的两路观测；图像位于该控制步执行前，
缓存步观测仍仅为诊断图，是否实际模型输入需看observation_used_for_inference。
服务端日志和握手信息继续保留；同一运行服务的采样随机状态可能已推进，不据此做严格增益因果比较。

运行结束后（包含正常生成诊断的受控停止）：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add segmented bounded OpenPI approach results"
git push origin main
```

116项CPU测试通过，覆盖60步预算、3个分段、缓存块中停止、进展不足、远离、近物体初始拦截、
预览图和运行器开关路由。实际模型轨迹和Isaac接口仍待Spark运行验证。

## 块长5：最多400步长程接近测试

60步结果0c00f3f60652中，三个分段均有进展，TCP到物体中心距离从366.33mm降至322.16mm，
最终水平偏差约5.48mm，仍高于物体中心约322.12mm。本轮延长模型接近时间，检查能否到达80mm停止边界。

新增显式 `--openpi-approach-long`，必须同时指定 `--openpi-approach`；只有openpi-step、块长5可用。
最多400个50ms控制步、80次推理、2000条10ms物理记录；每20步仍检查进展。
长程模式总位移边界为350mm、相对初始姿态的净旋转边界为90度。
每步1mm/0.25度限幅、夹爪强制打开、物体移动与关节等原有检查保持有效。
不传长程开关时，原接近模式仍最多60步，边界仍为70mm/18度；普通模式仍最多20步。

继续使用8000端口的模型服务及其启动日志，在另一个终端运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_j13_candidate_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 5 --openpi-steps 400 \
  --openpi-approach --openpi-approach-long \
  --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" --record-physics
```

输出配置名称增加 `-approach-long`。任一步进入80mm边界，或距离相对初始增加超过5mm，
或某20步分段进展不足2mm，均提前停止。不要为执行满400步关闭这些检查。
进入80mm边界时 `approach_summary.near_boundary_reached=true`，
`termination_reason=near_object_boundary`、`stopped_by_monitor=true`、`probe_completed=false`。
这是本轮预期的受控停止，不是抓取成功；夹爪仍保持打开，也不保证400步内一定到达边界。
最多400步对应20秒仿真时间，推理及渲染会增加实际等待时间。

结束后导出并推送，包括提前受控停止的结果：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add long segmented OpenPI approach results"
git push origin main
```

121项CPU测试通过，包括400步路由、400步完整预算、块内进入80mm边界、延长后进展不足停止、
错误开关组合和原60步预算。Python语法检查通过；实际长程轨迹、接触情况和Isaac运行仍待Spark验证。

## 长程接近：相同初始化连续重复3轮

结果11e909446243中，320步将距离从366.33mm降至147.45mm，最近139.44mm；末段模型指令开始抬升，
触发20步进展不足停止。本轮先检查这种停滞是否重复出现，不改变控制器、动作限幅、图像或停止规则。

`openpi_approach_repeat_cases.json`包含repeat1/2/3三个名称，增益与`openpi_j13_candidate_cases.json`完全相同。
利用已有运行器，每个case启动一个新的Isaac进程，重建环境并以seed=0和固定方块位置初始化。
三轮顺序执行，分别保存trace、图像、执行器参数、服务启动日志和10ms物理记录，统一放入同一汇总。
不需要修改运行器或策略执行代码。

三轮复用同一OpenPI服务，服务内采样随机状态可能持续推进；这是相同物理条件的顺序重复，
不是三个有独立受控模型随机种子的统计样本。运行后还需比较三轮初始state和两路图像，确认输入是否相同。
不要在轮次之间重启服务、变更prompt或操纵场景。

保持8000端口模型服务运行，另一终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_approach_repeat_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 5 --openpi-steps 400 \
  --openpi-approach --openpi-approach-long \
  --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" --record-physics
```

每轮最多400控制步/80次推理/2000物理子步；三轮合计最多1200步/240次推理/6000物理记录。
仍强制开爪，每步1mm/0.25度，总位移350mm/净旋转90度，接近80mm或进展不足等条件仍提前停止。
正常生成诊断的受控停止会保留结果并继续下一轮；进程异常或报告缺失则运行器停止，应保留报错排查。
Viser随每个Isaac进程退出并重启，轮次切换时页面可能断开，刷新即可；无需为浏览器断开重跑已完成轮次。

重点比较：初始输入一致性、最近/最终距离、停止步数及原因、最后20步模型请求与实际Z运动、
水平偏差、周期末TCP跟踪误差、10ms估算力矩和图像可见性。
三轮进入80mm边界的次数只用于这组条件的描述，不等于抓取成功率或泛化能力。

完成全部三轮后一次性导出并推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add three repeated long OpenPI approach results"
git push origin main
```

本轮配置通过增益校验及名称长度/唯一性校验；用模拟子进程验证三轮路由、受控停止后继续执行、
三份trace及预览图无覆盖导出，9项现有运行器测试通过。实际环境重置一致性及三轮模型结果待Spark验证。

## 冻结观测：同一输入的完整动作块与限幅核对

三轮结果2397bb6005eb均在126.9–144.1mm最近距离附近触发进展不足停止。
本轮不再移动机械臂，直接读取这三轮保存的模型输入，检查相同输入下动作的变化以及限幅影响。

新增 `frozen_observation_probe.py`，不启动Isaac、不创建环境、不执行机械臂或夹爪动作。
选择4组观测：第一轮初始观测，以及三轮最后一个控制步执行前的观测。
最后观测可能原本属于块内缓存步，当时没有送给模型；本轮明确将该保存观测作为新模型输入，
不能将新返回的动作与原缓存动作当作同一输入的重现。

每组连续查询3次，共12次推理。原始224×224 RGB PNG不再缩放、旋转或修改，状态使用保存的
`state_observed`转为float32，prompt使用原运行的值；记录每次输入摘要以检查同组输入一致性。
模型服务的随机状态继续推进，没有控制模型随机种子，也不模拟执行动作后的未来观测。

运行前保持8000端口模型服务启动。以下命令固定读取已分析的三轮源目录，避免latest指向别的测试：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only

cd /home/test/Robot/IsaacLab
uv run --extra viser --extra openpi python \
  /home/test/Robot/Cart-Double-Passive-Direct/openpi_isaac_bridge/frozen_observation_probe.py \
  --latest /home/test/Robot/IsaacLab/logs/openpi_gain_sweep/20261029T092353347485Z/sweep_summary.json \
  --repeats 3 \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log
```

使用Lab环境的openpi/viser依赖，但本脚本不需要SimulationApp，且不会启动Viser画面。
原三轮目录中的PNG需要保留，缺失时脚本会报错，不会换用其他帧。

输出自动保存在仓库 `diagnostics/openpi_frozen_observation/<UTC时间>/`，每次使用新目录：

- `report.json`：源清单SHA256、源步数、状态、prompt、图像SHA256、服务握手信息、12次完整动作块。
- 每个动作的原始值、已有LIBERO映射解码值、假设应用1mm/0.25度限幅后的值及保留比例。
- 分别汇总前5个动作和整个动作块的平移向量和、固定观测方向上的接近投影、限幅次数及闭爪请求数。
- 8张原样复制的观测PNG，以及用户指定服务启动日志的最多128KiB前缀。

`[Frozen query]`中接近投影为正表示朝向方块，为负表示远离；bounded Z正值表示向世界+Z抬升。
这些是固定观测下的数值汇总，不是执行轨迹、IK目标或接触结果。物体真值只用于投影诊断，不送入模型。
映射记录中的夹爪值也只是数值对照；每项executed=false，报告actions_executed=0。

报告保留源TCP偏置和坐标假设；实际相机外参/内参、LIBERO训练TCP注册资料不在源清单中，明确标为未验证。
这轮不会声称已校准相机或TCP，也不会基于缺失资料修改坐标变换。

正常完成时report.status=completed、4组各3次查询；失败时保存已完成的查询及错误，不应当作完整结果。
本地128项CPU测试与语法检查通过；真实保存图像配合模拟服务的端到端检查确认4×3查询、重复输入相同、
8张图无修改复制、JSON保存且没有Isaac导入。实际OpenPI响应仍待Spark验证。

结束后直接提交新目录，不需要调用gain_sweep的导出脚本：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git add -- diagnostics/openpi_frozen_observation/
git commit -m "Add frozen observation OpenPI action diagnostics"
git push origin main
```

## 同一动作块：前5步与完整10步的等时重放

冻结观测结果dc5dcaefdce0中，12个动作块的后5步均有正接近投影，前5步有7次负投影。
本轮检查截断块对实际运动的局部影响，不重新调用模型，也不改变增益或每步限幅。

固定使用初始观测第3次查询的完整10步动作，写入`openpi_chunk_prefix_actions.json`。
该块前5步固定方向投影-0.662mm，完整10步+3.279mm，是特意选择的诊断反例，不是随机样本或成功率测试。
fixture保存源提交、报告SHA256、输入SHA256、查询编号和原初始TCP位姿，动作值原样保存。
只在初始姿态做对照，不声称重现近物体关节状态。

两组分别启动新Isaac进程、固定seed=0和方块位置，保持候选增益及夹爪打开：

- `openpi-replay-prefix5`：执行动作0–4，然后5步零平移/零旋转增量。
- `openpi-replay-full10`：执行动作0–9。

两组均为10个50ms控制周期，共0.5秒仿真时间、50条10ms物理采样。前5步动作完全相同。
零输入尾段每次相对于实测当前TCP生成零增量，不是锁定第5步目标的定点保持；惯性和控制响应仍可能带来位移。
每步仍为1mm/0.25度范数上限，夹爪始终打开，原重放总位移30mm/净旋转6度及单周期等检查保持有效。
初始化TCP必须与源记录相差不超过0.25mm/0.1度，否则在执行前停止；两组初始关节位置/速度也保存用于事后比较。
原来的三种20步分量重放仍为20步动作+20步零输入；新10步fixture只能与新两种模式配套使用，混用会拒绝。

不需要启动OpenPI服务，执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_j13_candidate_cases.json \
  --modes openpi-replay-prefix5 openpi-replay-full10 \
  --replay-actions-json openpi_isaac_bridge/openpi_chunk_prefix_actions.json \
  --record-physics
```

Viser会随两个进程分别启动，但每组运动很短，应以保存记录为准，不为观看增加物理步或重复执行。
预期每组replay_result.probe_completed=true、completed_steps=10；物理记录recorded_controls=10、recorded_substeps=50。
prefix5的active_steps=5、zero_input_steps=5；full10分别为10和0。
物理记录phase也区分replay/zero-input-tail。报告的comparison_window_error_summary覆盖相同10周期，
period_error_summary只覆盖各自实际动作段，前者更适合等时误差对照。

比较两组初始TCP和关节状态、前5步响应是否一致、第5/10步位置与物体距离、后5步实际位移、跟踪误差及估算力矩。
即使full10更接近，也只能证明本固定块的局部差异，不能直接推断长程闭环或抓取成功。

运行完成后导出、推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add fixed chunk prefix5 versus full10 replay results"
git push origin main
```

133项CPU测试及语法检查通过，覆盖等时计划、共享前5步、零输入尾段、完整未来动作校验、fixture混用拒绝和初始化不匹配拦截。
模拟子进程检查确认两组独立启动、同一fixture/候选增益、无需OpenPI依赖、两份物理轨迹无覆盖导出；实际Isaac轨迹待Spark验证。

## 长程模型闭环：块长10，重复3轮

固定块重放结果dd3dcdc03575中，两组初始状态和前5步响应完全相同；完整10步净接近3.500mm，
前5步加零输入尾段净远离0.575mm。本轮进一步检查真实模型反馈下保留后半段动作的实际表现。

新增执行块长10选项，复用同一组repeat1/2/3候选增益配置。每次推理必须返回至少10个有效动作，
按顺序执行索引0–9，然后使用新图像和新状态重新推理。若模型以后返回超过10步，只执行前10步，不假定任意长度全部执行。
每个50ms控制步仍采集诊断观测并检查状态和停止条件；块内观测不送给模型，直到下次块边界。
停止条件在块内也有效，不会为了执行满10步而忽略停止。

增益、seed=0、固定物体位置、图像处理、世界坐标状态、每步1mm/0.25度限幅、强制开爪和监测保持不变。
每轮最多400步/40次推理/2000条10ms物理记录；每20步至少接近2mm，距离进入80mm或相对初始远离超过5mm等情况提前停止。
总位移350mm/净旋转90度和原单周期、物体移动、软关节限位检查继续有效。
每10步重新推理对应0.5秒仿真执行窗口，不是实时反馈频率保证；推理期间仿真暂停。

保持原OpenPI服务在8000端口运行，另一终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_approach_repeat_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 10 --openpi-steps 400 \
  --openpi-approach --openpi-approach-long \
  --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" --record-physics
```

三轮分别新建Isaac进程，名称包含`repeatN-chunk10-approach-long`，不会覆盖chunk5结果。
受控停止后自动运行下一轮。Viser进程切换时刷新页面即可，不要为观看改变控制流程。
三轮合计最多1200控制步、120次推理和6000物理记录；实际可能提前停止。

重点检查：三轮初始状态/图像、chunk_action_index是否0–9、推理是否只在第1/11/21等步发生、最近距离与停止原因、
前5和后5步的实际接近量、水平偏差、关节/TCP跟踪误差、估算力矩和物体位置。
不同模型查询仍存在输出变化，不能把本轮与此前chunk5轨迹的全部差异严格归因为块长。
进入80mm只代表达到诊断边界，夹爪仍打开，不代表抓取成功。

完成三轮后导出推送：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add three chunk10 long OpenPI approach results"
git push origin main
```

138项CPU测试和语法检查通过，新增覆盖10步完整顺序、短块拒绝、400步40次推理、块内监测停止、三轮名称与路由。
原块长1/5和固定动作重放测试继续通过，实际长程模型轨迹待Spark验证。

## 块长10：仅将预算增加到500步，重复3轮

结果472997a7ce83中，第一轮执行满400步，末段仍接近6.70mm，最终83.77mm；尚未进入80mm边界。
本轮只增加可执行时间，保留所有运动和停止约束，不同时调整进展门槛、增益、图像或动作幅度。

长程模式现在允许显式`--openpi-steps 500`；原400步命令仍只执行最多400步，普通模式20步、非长程接近60步上限保持不变。
新增`openpi_approach_500_cases.json`，三组增益与此前repeat配置相同，名称加入budget500，防止导出覆盖400步实验文件。

每轮最多500个50ms控制步（25秒仿真时间）、50次推理、2500条10ms物理记录。
三轮合计最多1500控制步、150次推理、7500条物理记录。每10步更新模型输入，每20步仍检查至少接近2mm。
夹爪保持打开，每步1mm/0.25度、总位移350mm/净旋转90度、80mm停止边界、远离与物体移动等检查均保持有效。
增加预算并不会保证执行满500步；任何停止条件触发仍立即终止该轮并保存诊断。

保持OpenPI服务在8000端口运行，另一终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_gain_sweep.py \
  --isaaclab-root /home/test/Robot/IsaacLab \
  --gain-cases-json openpi_isaac_bridge/openpi_approach_500_cases.json \
  --modes openpi-step \
  --openpi-replan-steps 10 --openpi-steps 500 \
  --openpi-approach --openpi-approach-long \
  --openpi-state-frame world \
  --server-host 127.0.0.1 --server-port 8000 \
  --policy-server-log /home/test/Robot/openpi/openpi_chunk_probe_server.log \
  --prompt "pick up the cube" --record-physics
```

重点看是否触达80mm边界，以及第400步之后还保持接近还是开始反向；同时核对三轮初始输入、
停止原因、近物体横向偏差、周期末跟踪误差和10ms估算力矩。不同查询的模型输出会变化，
本轮不是从上一轮400步末状态接着运行，也不保证复现上一轮第一条轨迹。

进入80mm边界时near_boundary_reached=true、stopped_by_monitor=true、termination_reason=near_object_boundary，
probe_completed=false是预期受控停止。执行满500步但仍未进入边界时probe_completed=true，仅表示预算执行完。
本轮依然只验证开爪接近，不评价抓取成功。

完成三轮后导出推送，包括正常生成诊断的受控停止：

```bash
python3 openpi_isaac_bridge/export_gain_results.py --include-physics-traces
git add -- diagnostics/gain_sweep/
git commit -m "Add three budget500 chunk10 OpenPI approach results"
git push origin main
```

140项CPU测试与语法检查通过，新增覆盖500步/50次推理/25个检查段和延长预算后的进展不足停止；
模拟三轮路由与导出确认同一增益、500步/块长10、独立名称、受控停止后继续以及三份物理轨迹导出。
实际500步模型轨迹仍待Spark验证。

## 静态输入几何审计：相机、手部、TCP与投影

500步结果f21e2a40f594中，三轮在300–320步提前触发进展不足，新增预算未被使用。
本轮不再增加步数或调整驱动，记录此前缺失的实际几何数据，区分测量、坐标假设和训练参考缺口。

新增独立运行器 `run_geometry_audit.py` 和 `geometry-audit`模式，复用当前候选增益、seed=0、固定方块位置、
50ms控制周期、两路相机及固定腕部初始瞄准方式。仅执行原有30+5步开爪零增量预热，
之后不执行诊断运动、不请求模型、不修改相机或TCP参数；采集前后物理步计数必须相同。
不用启动OpenPI模型服务；openpi依赖仅用于复用相同的224像素图像预处理。

执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_geometry_audit.py \
  --isaaclab-root /home/test/Robot/IsaacLab
```

结果直接写入仓库 `diagnostics/geometry_audit/<UTC时间>/`，无需gain_sweep导出：

- `geometry_audit.json`：基座、传感器父刚体、IK父刚体、传感器TCP、IK TCP与物体根坐标；配置偏置、关节位置/速度、模型8维世界状态。
- 相机实际K、ROS/OpenGL/world朝向、双向4×4变换、帧号、拍摄元数据及原图摘要；不从配置中的eye/target推测实际相机位姿。
- 原始256像素PNG、原预处理生成的224像素PNG、两张`*_projection.svg`独立投影图。SVG嵌入原始PNG字节并叠加矢量标记，不修改模型输入图像。
- TCP/IK偏置闭合：父刚体实测位姿乘配置偏置，与对应传感器/控制器实测位姿比较。0.1mm/0.05度标记只评价这项内部闭合。
- `camera_implementation.txt`：已安装运行时相机属性的源代码，便于核对实际版本；`actuator_audit.json`、`source_case.json`、`joint_gains.json`、`run.json`和控制台日志。

投影使用ROS相机坐标：+Z前方、+X向右、+Y向下，u/v由K和相机坐标计算。
约定来自[Isaac Lab相机数据接口](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/camera/camera_data.py)。
图中标记物体根坐标、传感器TCP、IK TCP、传感器父刚体和机器人根坐标，behind/outside情况记录到JSON；
inside_image仅表示位于视锥内，不处理遮挡。物体根坐标和手指刚体原点也不冒称表面中心或接触点。

若RGB包含绑定的capture位姿/内参，则用于投影；否则明确使用当前CameraData位姿并标记拍摄时刻绑定不可用。
刚体点均为当前静态采集位姿，未提供延迟图像的刚体历史；该限制也保留在报告中。
没有人工或独立检测的2D对应点，本轮不自动给出像素重投影准确率；投影图用于后续检查。

此模式的observation.json也保存与闭环相同语义的世界TCP状态、带符号旋转向量和[q1,-q2]，不使用旧捕获模式的两指同号状态。
姿态四元数为当前Isaac运行时xyzw顺序；模型状态适配遵循[OpenPI LIBERO示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。

**解释边界：**内部偏置闭合通过不代表LIBERO训练TCP已对齐。训练环境的物理安装关系、相机参考和跨场景注册没有独立测量，
仍标记unverified，不会自动修改偏置、图像翻转或动作映射。本轮也不验证抓取或策略成功。

正常完成run.json和geometry_audit.json均为completed，policy_queries=0、diagnostic_motion_steps=0，
采集前后physics_step_count相同；失败时保留日志和部分报告，应先检查console.log。
147项CPU测试及语法检查通过；模拟采集验证当前位姿/绑定capture元数据两种路径、偏置闭合、状态符号、图像/投影文件以及无额外物理步。
实际Isaac相机与刚体数据接口仍待Spark验证。

运行结束后直接推送：

```bash
git add -- diagnostics/geometry_audit/
git commit -m "Add static camera and TCP geometry audit results"
git push origin main
```

## LIBERO参考侧：依赖检查与同关节姿态的静态采集

Isaac审计b775a025b12e显示内部偏置闭合与图像投影一致；尚缺训练参考侧的EEF和相机记录。
新增`libero_reference_audit.py`，读取最新的已完成Isaac几何审计，先检查现有Python环境，条件具备时采集LIBERO参考。
不启动模型服务、不查询策略、不安装包、不改变Isaac配置，也不修改当前OpenPI环境。

按[OpenPI LIBERO示例说明](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)，
模型服务和LIBERO仿真可以使用不同环境。本脚本优先检查`openpi/examples/libero/.venv/bin/python`，
再检查`openpi/.venv/bin/python`和运行脚本的Python；可用`--python /绝对路径/python`指定已有解释器。
检查只读取模块位置和发行版本，包含numpy、PIL、yaml、mujoco、robosuite、libero、openpi_client和torch。
运行时只在子进程的PYTHONPATH中加入现有`third_party/libero`，不会初始化/下载子模块。

执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/libero_reference_audit.py \
  --openpi-root /home/test/Robot/openpi
```

默认从仓库`diagnostics/geometry_audit/`选取最新已完成记录；需要固定源时，可追加：

```text
--isaac-audit /home/test/Robot/Cart-Double-Passive-Direct/diagnostics/geometry_audit/20261030T021135995628Z/geometry_audit.json
```

输出直接写入`diagnostics/libero_reference/<UTC时间>/`。状态必须按以下含义解释：

- `dependency_missing`：现有候选解释器没有完整依赖，只保存可用性/版本报告；没有生成LIBERO几何参考。这是预检完成，不是几何测试通过。
- `completed`：原生LIBERO环境创建、静态关节设置、观测和相机采集完成。
- `capture_failed`或`capture_timeout`：依赖表面可用但实际导入、资源或渲染等失败；查看`reference.json`和`worker.log`。原生采集子进程有120秒时限。

依赖不齐也请保留并推送报告，下一步按实际解释器、架构和缺失项准备环境，不直接套用旧版CUDA/torch锁定文件。
运行器不会自动安装依赖或替换模型服务环境。模型服务可以继续运行，本脚本不连接其端口。

条件具备时，默认采集`libero_spatial`第0个任务、seed=0、256像素双相机；使用普通任务reset，
不加载演示初始状态、不评测任务成功率。默认使用子进程EGL离屏渲染，后端信息记录到报告。
LIBERO首次导入的路径配置写在本次结果目录的`libero_config/`中，并通过子进程LIBERO_CONFIG_PATH使用，
不改写`~/.libero`。接口依据[LIBERO环境包装器](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)核对。

静态参考采集内容：

- 依据实际关节名后缀匹配7个臂关节和2个夹爪关节，检查MuJoCo关节范围后设置为Isaac源角度；夹爪采用参考侧[q1,-q2]符号。
- 只直接设置qpos、将相关qvel置零并调用forward；不调用env.step、不执行dummy动作或模型动作。采集前后MuJoCo时间必须相同。
- 保存机器人根刚体、EEF手部刚体、夹爪根刚体、实际grip_site的局部/世界位姿，以及观测中的EEF位置与四元数。
- 分别比较观测位置与grip_site、观测朝向与EEF刚体/站点朝向，保存已安装观测代码，避免混淆参考框架。
- 保存agentview与robot0_eye_in_hand的实际位置、朝向、父刚体、fovy和推导K，以及世界/基座参考下的记录。
- 保存观测原图256、按官方示例旋转180度后的256图及模型224图，共6张PNG。此处只处理LIBERO参考图，不改变Isaac图像。
- 保存实际机器人/夹爪/合并场景XML、控制器限幅参数、源代码、版本预检与Isaac源记录摘要，便于核对依赖和物理安装关系。

观测与预处理依据[OpenPI LIBERO示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)；
[robosuite 1.4.1观测实现](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/robots/single_arm.py)中，
EEF位置取夹爪站点，四元数取EEF手部刚体；脚本保留两种来源及实际安装版本代码，不假定它们局部朝向相同。

解释边界：直接设置相同数值的关节角不代表驱动已经到位，也不能自动证明关节零点/符号、基座、手部框架和场景已注册。
本轮场景不是Isaac的方块场景，参考环境版本也未独立证明与checkpoint训练版本完全相同；这些差异保留为待比较项。
不会据局部偏置数字自动替换Isaac TCP或翻转其图像。

154项CPU测试与语法检查通过；模拟原生接口采集覆盖9关节设置、6种图像输出、EEF来源、XML保存和无env.step/时间推进。
实际LIBERO依赖、资源和GPU渲染可用性仍待Spark运行确认。

无论预检缺依赖还是采集完成，结束后推送本轮记录（不调用gain_sweep导出）：

```bash
git add -- diagnostics/libero_reference/
git commit -m "Add LIBERO reference availability and geometry audit"
git push origin main
```

## 依赖预检e8840bdf84a8之后：独立参考环境安装并重试采集

实际报告显示：OpenPI环境为Python3.11.17/aarch64，已有numpy1.26.4、mujoco2.3.7、torch2.7.1和LIBERO资源，
首轮模块预检缺robosuite；`examples/libero/.venv`不存在。系统Python3.12缺少更多依赖。
这只说明模块可定位，不代表其他依赖已经成功导入或GPU渲染可用。

新增`setup_libero_reference_env.py`，创建`/home/test/Robot/openpi/examples/libero/.venv`，与模型环境`openpi/.venv`分开。
只借用现有Python3.11解释器创建新venv，不启用system-site-packages、不复用模型环境包目录；所有uv pip install均显式指定新解释器。
不运行OpenPI的uv sync、不修改其锁文件，也不安装到系统Python。已有未由本脚本管理的同名环境会拒绝覆盖，不自动删除。

直接运行，安装完成后自动执行同关节姿态的参考采集：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/setup_libero_reference_env.py \
  --openpi-root /home/test/Robot/openpi
```

首次需要下载并安装参考依赖，可能耗时数分钟，终端会输出各阶段进展。
直接运行时依赖方案见`libero_reference_runtime.txt`：固定Python3.11对应的numpy1.26.4、mujoco2.3.7、robosuite1.4.1、torch2.7.1、
numba0.61.2、scipy1.15.3和必要的静态环境导入依赖；这是参考采集的运行时配方，不是checkpoint训练环境的完整复现或完整哈希锁。
robosuite版本与[OpenPI LIBERO示例依赖](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/requirements.in)一致；
已通过PyPI官方元数据核对MuJoCo、PyTorch、Numba/llvmlite、SciPy、OpenCV、Matplotlib的相应ARM轮子。

安装阶段先装显式运行时，再在新环境中安装本地openpi-client及其依赖，最后以`--no-deps`安装已有LIBERO源码，
避免LIBERO旧训练依赖列表重新强制安装旧numpy/torch等。本配方只支持这项静态几何采集，不宣称支持全部LIBERO训练流程。
LIBERO源码和assets继续使用现有third_party目录，不重新下载数据集。

离屏子进程使用EGL，并将PYNPUT_BACKEND设为dummy用于无桌面输入导入；不启动键鼠监听器、不修改主进程桌面环境。
后端选择依据[pynput实现](https://github.com/moses-palmer/pynput/blob/v1.7.7/lib/pynput/_util/__init__.py)。

输出保存在仓库`diagnostics/libero_reference_setup/<UTC时间>/`：

- `setup.json`和各阶段日志：命令、返回码、安装/采集状态。
- `model_environment_before.json`及`model_environment_after.json`：原模型环境的Python和发行包版本清单，用于检查本次前后是否相同；这不检查所有文件或并发进程状态。
- `runtime_requirements.txt`：本次实际使用的直接版本配方副本。
- `capture/<UTC时间>/`：自动重试的预检、reference.json、图像、XML及worker日志。

状态`completed`只在安装和参考采集均完成时出现。`environment_ready_capture_failed`表示新环境已安装，原生导入、资源或渲染等仍有问题，
应保留capture子目录日志。安装阶段失败则为failed，停止后续步骤并保留安装日志；不会回退到给模型环境补装。
可用`--skip-capture`只安装新环境；正常测试命令无需加该选项。

158项CPU测试和语法检查通过，覆盖安装目标、新环境保护、模型版本前后记录以及不能将缺依赖采集标成成功。
ARM轮子元数据已核对，但尚未在Spark实际解析安装全部传递依赖、编译潜在源包或验证EGL；真实结果以日志为准。

运行结束后，推送整个安装及参考采集记录，无需再次手工启动审计或运行gain_sweep导出：

```bash
git add -- diagnostics/libero_reference_setup/
git commit -m "Add isolated LIBERO setup and reference capture results"
git push origin main
```

## 下一轮：冻结图片的坐标 A/B/C 对照

参考采集结果 `1f301b39520b` 已成功。此轮仅检查模型对状态坐标变化的敏感性，不启动 Isaac，不执行动作。
使用先前四个冻结观测，逐一保持两张224×224 RGB图像、提示词、姿态编码、夹爪状态不变：

- A：原始Isaac状态。
- B：假设世界轴对齐，位置加上 `[-0.66, 0, 0.912]` m。
- C：在B基础上，加上 `R_hand * [0,0,-0.0064]` m，将103.4 mm传感器点换成97 mm参考点。

姿态使用手部旋转向量，保持原值，不换成grip_site姿态。参考点转换随每张观测的手部姿态变化。
该变换只是单姿态几何比较得到的候选，不是多姿态标定结果。摄像机与场景差异依然存在。
每个观测每条件查询5次，共60次；每轮轮换条件顺序，记录随机采样变化，没有共享随机种子或配对噪声保证。
记录完整原始10步、前5步与全部动作的方向投影、假想限幅结果和闭爪请求；投影中物体真值只做诊断，不发送给模型。
物体诊断坐标与状态坐标使用相同世界平移；C额外改变测量点。正投影不能证明真实抓取或控制成功。

保持原有OpenPI服务运行，在另一个Spark终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/.venv/bin/python \
  openpi_isaac_bridge/frozen_coordinate_probe.py --repeats 5
```

脚本自动选择仓库 `diagnostics/openpi_frozen_observation/*/report.json` 中最近完成的四观测报告。
可用 `--source-report /绝对路径/report.json` 明确指定。
查询前检查源状态103.4 mm偏移、世界帧、原点附近的基座、旋转向量编码、图像SHA及尺寸。
候选使用同一证据commit，源报告SHA、原始setup、实际发送float32状态、每次输入SHA都会保存。
输出 `diagnostics/openpi_coordinate_probe/<UTC>/report.json`，失败也保留已完成查询。`actions_executed` 始终为0。
本地164项CPU/模拟客户端测试通过；Spark真实模型结果待本轮运行。

完成后直接推送本轮目录，无需运行gain_sweep导出：

```bash
git add -- diagnostics/openpi_coordinate_probe/
git commit -m "Add frozen coordinate A B C comparison results"
git push origin main
```

## 下一轮：11姿态静态几何标定与留出验证

冻结A/B/C结果21259921e8b9显示世界坐标转换值得继续检查，TCP转换没有一致行为改善。
本轮先验证几何，不调用OpenPI服务、不执行模型动作、不改刚度阻尼、不做抓取。

`run_multipose_geometry.py`依次启动11次独立Isaac场景：以原几何审计实际关节角为基准，
7个拟合姿态（原位及J1到J6分别约0.15–0.20rad变化），4个未参与拟合的混合关节姿态。
每次仅更换初始关节配置、保持夹爪张开，沿用已有35步hold预热与静态采集。
每个姿态独立初始化，Viser可能断开后重连，不是连续关节运动轨迹。
随后用已安装的独立LIBERO环境，直接设置对应Isaac采集到的实际7关节及双指位置，forward后采集。
用实际位置对照，不以名义初始角度代替实际角度。每对来源SHA、命令和日志均保存。

分析先用7个手部原点拟合世界刚体变换，再分别计算原103.4mm TCP、固定转换97mm TCP的误差；
额外仅用拟合组估计三维手部局部TCP修正，供诊断比较。4个留出姿态都不参与拟合。
记录位置RMS/最大值、手部朝向误差、实际关节匹配误差与名义初始角偏差。
拟合点第二奇异值不足5mm则停止，避免退化位置分布产生看似可信的标定。
诊断阈值为留出姿态校正TCP最大误差1mm、手部朝向最大误差0.1度；它们是本轮自定阈值，不是行业标准。
`completed`表示采集分析完成；`candidate_geometry_within_thresholds`才表示本次新拟合变换满足这两项阈值。
这不验证原先固定[-0.66,0,0.912]平移必然最优，不验证相机、动力学或抓取性能。

在Spark仓库终端运行（无需启动OpenPI服务，现有服务可保持运行）：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_multipose_geometry.py
```

自动读取原 `diagnostics/geometry_audit/` 最近完成的审计作为设计基准。
可用 `--source-audit /绝对路径/geometry_audit.json` 指定。
使用现有 `/home/test/Robot/openpi/examples/libero/.venv/bin/python`，不安装任何包。
需要多次启动仿真，预计比单姿态采集耗时长；每个姿态会显示阶段进度。
结果保存为 `diagnostics/multipose_geometry/<UTC>/`，含11组原始审计、图片、XML、日志及最终analysis.json。
失败会停止并保留已完成姿态，推送整个目录即可诊断。169项本地CPU/模拟流程测试通过；真实Spark采集待运行。

完成后：

```bash
git add -- diagnostics/multipose_geometry/
git commit -m "Add multi pose geometry calibration and held out validation"
git push origin main
```

## 多姿态首轮1383effb9bdf：解释器符号链接修复

首轮没有完成11姿态：home的Isaac审计成功，传感器闭合误差约1.33e-8m，IK闭合误差约6.22e-8m；
随后LIBERO停在dependency_missing预检，未创建参考环境，也没有最终analysis.json。
run.json实际使用 `/home/test/.local/share/uv/python/cpython-3.11.17-linux-aarch64-gnu/bin/python3.11`，而不是隔离参考venv入口。
原因是runner对venv/bin/python调用Path.resolve()，展开了符号链接，导致启动基础解释器、失去venv包搜索路径。
这次日志不是TCP或雅可比误差证据，也不证明原venv的包丢失。

修复后保留venv入口路径，只展开用户目录并转换成绝对路径，不解析符号链接。
新增启动前预检：在任何Isaac启动前记录实际sys.executable、sys.prefix、sys.base_prefix和8个模块可定位性；
若选到基础解释器或缺模块立即停止，不先浪费一次Isaac采集。
预检只读取模块位置，不加载原生仿真；真正导入与渲染仍由各姿态LIBERO采集验证。
不会安装包、修改模型环境或覆盖旧报告。失败姿态现在明确标记failed。
172项本地CPU/模拟流程测试通过，新增解释器路径、基础解释器拒绝、缺模块及启动前停止回归。

拉取修复后重跑同一设计，明确指定现有隔离环境入口：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_multipose_geometry.py \
  --reference-python /home/test/Robot/openpi/examples/libero/.venv/bin/python
```

生成新的UTC目录，保留首轮失败记录，不需要重新安装依赖。
完成或失败后推送整个结果目录：

```bash
git add -- diagnostics/multipose_geometry/
git commit -m "Retry multi pose geometry with preserved reference venv"
git push origin main
```

## 下一轮：复用11姿态的相机元数据与图像方向审计

`camera_reference_audit.py`读取已完成的多姿态目录和analysis.json，不启动Isaac或LIBERO，不查询模型，不更改运行相机。
逐对验证原始审计SHA、两侧RGB原图SHA、LIBERO180度旋转的像素关系，生成22组相机对照。
核对配准后的光心、完整朝向与光轴夹角、内参/FOV，以及手腕相机相对手掌的安装变化。

已用f6b92772cdc7的11组真实采集在本地完整运行：Isaac手腕安装位置基本固定，但跨初始化姿态的安装朝向相对home最大变化约89.594°；LIBERO局部朝向保持固定。
现有Isaac初始化代码每次朝世界参考点重新对准相机，因而不能将这些不同初始姿态的相机视作同一固定安装。
这不是同回合相机抖动或导致策略停滞的证据：同样初始姿态的回合仍可能共享相同安装。
外部光心位置也有约837mm差异、光轴约47.7°，视场角两边并不相同，不能仅靠翻转图片完成对齐。

直接使用现有独立参考环境执行，无需启动任何仿真或模型服务：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/camera_reference_audit.py
```

默认读取最近完成的 `diagnostics/multipose_geometry/*/run.json`；可用 `--source-manifest /绝对路径/run.json` 指定。
输出 `diagnostics/camera_reference_audit/<UTC>/`：

- `analysis_zh.md`、`report.json`：几何差异与每姿态投影。
- `comparison.html`：原始/180度旋转图，22组四列画廊，可浏览器打开。
- 每姿态子目录：源RGB副本、Isaac旋转副本、投影SVG，不覆盖原始采集。
- `candidate_camera_parameters.json`：参考外部相机映射到Isaac世界、参考手腕相机的固定手部局部安装、256像素内参，仅为建议参数，未应用。

红色投影是从仿真三维坐标算出的标签，不是图像检测结果；可能被遮挡。
原生帧缓冲方向还没有用独立可见标记验证，因此既不自动选择最佳翻转，也不宣布相机已标定。
下一阶段才在静态采集里应用候选光学参数，并用可见标记核对渲染像素。
相机ROS光轴约定与接口参考[Isaac Lab官方相机文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/api/lab/isaaclab.sensors.html)；
本轮四元数顺序沿用实际保存的xyzw审计，不借用旧版文档的四元数排列假设。
176项本地CPU测试通过，完整流程另已通过已下载的真实11姿态资料验证；Spark复现后推送：

```bash
git add -- diagnostics/camera_reference_audit/
git commit -m "Add camera geometry and image orientation audit"
git push origin main
```

## 下一轮：候选相机在Isaac中的静态渲染标记验证

`run_camera_alignment.py`读取最近完成的相机审计，校验源manifest SHA，复制candidate_camera_parameters.json，
依次为home、mixed34、mixed56创建3次独立静态采集。使用旧Isaac审计实际关节角初始化，不调用OpenPI，不改关节增益。
仅显式geometry-audit测试选项支持相机候选，普通训练、播放和模型动作流程不变。

在35步hold预热的最后5步之前，将外部相机设置到候选Isaac世界位姿；手腕相机按照当前手部位姿安装候选局部变换，
覆盖旧look-at朝向，并更新参考内参。后续不再朝世界目标重新对准。
使用当前[官方Camera实现](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/camera/camera.py)的set_world_poses（ROS、xyzw）和set_intrinsic_matrices接口。
实际安装后的位姿/K与setter实现源码保存，便于发现本地版本差异。

每台相机前方生成3个不对称分布的品红、青、黄球形标记，发光材质，不添加刚体或碰撞API。
标记世界坐标由事先请求的相机参数推导，期望像素为(58,67)、(184,91)、(103,192)。
随后检测真实RGB里的颜色连通区域质心，对比原图、水平翻转、垂直翻转和180度旋转四种假设。
这比只将三维点投影回元数据更进一步，但颜色检测仍是启发式；结果需结合原图检查遮挡/漏检。
这些标记属于独立标定场景，绝不能用作策略输入图像。

运行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_camera_alignment.py
```

无需启动OpenPI或LIBERO服务，Isaac会启动3次；Viser可查看标记场景。使用现有候选关节增益。
输入默认最近完成的 `diagnostics/camera_reference_audit/*/report.json`，也可用`--camera-audit /绝对路径/report.json`指定。
输出 `diagnostics/camera_alignment/<UTC>/`，包括：

- 顶层run.json：3姿态汇总、相机固定安装变化、图像方向一致性及all_camera_tests_passed。
- 每姿态capture子目录的camera_alignment_setup.json、camera_alignment_result.json、geometry_audit.json。
- 原始256RGB、4种方向副本、期望位置叠加SVG、全部颜色组件及匹配误差；候选参数、API源码与日志。

诊断阈值：位姿位置0.1mm、朝向0.05度、内参元素0.05像素；3标记最大误差2像素，最佳与次佳假设RMS差至少10像素；
跨姿态安装变化仍需小于0.1mm/0.05度且每台相机的图像方向一致。
完成采集不等于通过，查看all_camera_tests_passed。像素检测不通过仍保存其余姿态；进程异常则停止并保存日志。
此次只验证Isaac原始图像与ROS光学几何的关系，不直接决定Isaac应如何对应LIBERO策略预处理，也不验证抓取。
182项本地CPU/合成图像/模拟调度测试通过；本机没有Isaac原生渲染，真实材质、标记可见性和GPU接口仍待Spark运行。

结束后（成功或失败均保留记录）：

```bash
git add -- diagnostics/camera_alignment/
git commit -m "Add static aligned camera landmark render results"
git push origin main
```

## 下一轮：LIBERO初始化布局与原生渲染方向

`libero_scene_probe.py`在现有独立LIBERO环境内创建libero_spatial任务0，默认seed7、episode0，
不把Isaac角度写进LIBERO、不查询模型、不启动Isaac，也不修改普通桥接器或相机配置。
本轮分开保存：

- `default_reset/`：只执行环境reset的姿态与场景。
- 若本机任务.pruned_init存在且受限加载成功：`benchmark_initial/`和`benchmark_after10wait/`。
- `markers/`：在最后一份状态上插入相机标记，验证原生RGB与官方旋转预处理后的图像方向。

[OpenPI官方LIBERO示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
在reset之后使用任务初始状态，再执行10步 `[0,0,0,0,0,0,-1]` 等待物体稳定，之后才开始模型查询。
本轮复现初始化与等待部分，不开始模型推理；10步是非模型的控制器空动作，不是零物理时间。
之前的同关节几何审计用于验证映射，不等同于这个基准初始化过程。

原始/官方180度旋转/224策略尺寸图像分别保存；关节角、末端与物体根位置、桌面top站点位置、相机参数和物体投影均记录。
与已完成Isaac标记测试的home几何姿态比较，单独报告关节和末端位置差异；物体根原点不是共同抓取表面，Isaac桌面高度未测量不会由方块中心猜测。

可视标记以世界固定、group1、无碰撞geom加入XML，不增加关节或动态body；原模型状态完全恢复后才渲染，验证nq/nv、相机参数、模拟时间及状态没有被标记插入改变。
marker_test对原始256RGB和官方180度旋转256RGB分别测试四种方向，不提前假定LIBERO与Isaac应该采用相同翻转。
只在本地已有.pruned_init文件上使用torch的weights_only=True和NumPy数组/dtype受限白名单；失败记录load_failed，不回退到任意pickle执行，也不自动下载数据集。
因此completed不代表基准初始文件可用，须分别查看benchmark_initialization_available和marker_test.all_marker_views_passed。

运行（外层系统Python调度，自动选择现有参考venv，保留其符号链接入口）：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/libero_scene_probe.py
```

无需启动OpenPI服务。输出 `diagnostics/libero_scene_probe/<UTC>/`，包含run.json、report.json、预检、worker.log、三类图像、标记前后XML和源Isaac姿态/世界配准副本。
不会改~/.libero，参考配置文件仍写入本轮输出目录。
186项本地CPU、XML约束、初始状态验证和模拟子进程测试通过；本机没有原生LIBERO环境，真实初始文件加载、XML重建及GPU渲染待Spark验证。

结束后推送整个目录，失败记录也保留：

```bash
git add -- diagnostics/libero_scene_probe/
git commit -m "Add native LIBERO initialization and camera direction audit"
git push origin main
```

## c8c56fa7f87e之后：同步MuJoCo衍生量再校验标记相机

实际运行已成功受限加载50组pruned_init，保存3阶段布局和18张干净图像，完成10步等待。
随后原始state逐元素恢复成功，但手腕相机位置在XML重建forward后出现约1.564e-6m差异，触发atol=1e-8检查，尚未生成marker_test。
这不是缺依赖或初始状态缺失，也没有LIBERO原生图像方向的最终结论。

[MuJoCo 2.3.7模拟流程文档](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)
说明先做前向计算、再积分状态，中间结果可能滞后一个时间步。该机制与当前记录相符；需重跑观测同步后再判断，不把它直接当作已经证明的唯一原因。
修复在每次布局保存及生成标记之前显式sim.forward()和force_update观测，检查模拟state和time没有推进。
报告新增observation_synchronization，记录刷新前后相机差异；marker_camera_restoration保存XML重建前后两份相机参数、差异和原始1e-8容差。
没有放宽容差、没有增加等待步、没有改控制器或相机安装参数。
benchmark_initialization_available现在在标记测试前写入，即使后面失败也可明确知道初始化是否成功。
188项本地CPU/同步顺序/不推进时间回归测试通过；本机无MuJoCo原生环境，实际问题是否解决以Spark重跑结果为准。

重跑命令与上一轮相同：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/libero_scene_probe.py
```

结束后：

```bash
git add -- diagnostics/libero_scene_probe/
git commit -m "Retry LIBERO scene probe with synchronized observations"
git push origin main
```

## e48f599ef265之后：方向测量通过，修复末尾混合关节字典汇总

本轮同步刷新后，XML恢复前后两台相机的position/rotation/K差异均为0，原1e-8容差已通过。
两台相机标记测量也全部通过：LIBERO原始RGB到ROS需flip_y，官方180度预处理图到ROS需flip_x，最大标记误差约0.725像素。
上一轮Isaac原图到ROS为identity，因此在已测光学约定下，候选Isaac输入应测试水平翻转匹配参考策略画面，而不是直接复制180度翻转。
尚未部署任何策略预处理改动，也不能据此断言抓取失败完全由镜像引起。

原运行最终failed发生在纯布局汇总：ordered_joints对合并arm+finger字典查找时，finger_joint1/2也符合joint1/2结尾，被当成重复机械臂关节。
现在机械臂匹配显式排除finger_前缀；真实arm重名仍拒绝。对已推送三个真实布局进行顺序复核正确，新增混合字典回归，189项CPU测试通过。
原始failed记录保留，不改写成completed。拉取修复后重跑同一命令以生成正常完成的完整记录：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/libero_scene_probe.py
```

结束后：

```bash
git add -- diagnostics/libero_scene_probe/
git commit -m "Complete LIBERO scene probe with distinct arm and finger lookup"
git push origin main
```

## 下一轮：三组无标记代理场景，分离初始姿态与目标XY影响

`run_clean_scene_capture.py`读取最新完整LIBERO初始化报告及其对应相机候选，校验源文件SHA，生成3组独立Isaac静态采集：

1. baseline：旧Isaac实际初始角度、旧方块XY、已验证参考相机。
2. arm-only：LIBERO benchmark_after10wait的7个实际机械臂角度、旧方块XY、同一套相机。
3. arm-and-target-xy：同一组LIBERO角度，方块XY改为目标黑碗根原点通过逆世界变换得到的水平位置，同一套相机。

本轮以方块作为代理物体，不移植碗、盘子或LIBERO桌面，不宣称完整场景复现。
只复制目标水平位置；方块继续使用原配置的出生高度和支撑表面，不能将碗根原点的Z直接作为方块中心Z。
已用46c3f50bb810实际记录生成配置，第三组XY约(0.596548,0.202035)m，参考角度来源是实际等待后的关节而非默认reset。
夹爪沿用张开保持，未强行匹配参考等待中的细小双指位置差。

新增`--clean-scene-json`仅用于geometry-audit，禁止与其他初始姿态/标记相机输入文件组合。
普通训练、播放、OpenPI动作流程不变。相机应用复用上一轮已在Spark验证的接口，关闭标记生成；
每个独立进程在现有35步hold预热后保存原始256RGB及224策略尺寸图像，同时额外保存左右翻转224副本。
没有模型查询、没有模型动作，也没有部署图像预处理变化。

报告核对：初始机械臂角误差<=0.001rad、方块XY误差<=1mm、方块落稳Z相对基线变化<=2mm；
相机位置/旋转/内参沿用0.1mm/0.05度/0.05像素阈值。
另外记录方块中心投影、距画面边界余量、局部颜色饱和像素数。投影不等于可见性，颜色数只作为启发式提示，不宣称分割成功。
baseline原姿态可能视野差，它是对照；all_geometry_checks_passed不等于两路目标都可见，也不等于抓取成功。

使用现有独立参考venv执行外层配置生成与调度（需要其中的NumPy/Pillow，不安装包）：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_clean_scene_capture.py
```

无需启动OpenPI服务；会独立启动Isaac 3次，Viser可查看无标记场景。执行器增益不变。
输出 `diagnostics/clean_scene_capture/<UTC>/`：run.json、每组请求配置、完整geometry_audit与clean_scene_result.json、RGB/左右翻转副本、源证据及日志。
`comparison.html`并排展示各组无翻转/左右翻转图，并附LIBERO基准官方预处理的干净参考图。
194项本地CPU、合成画面、三组调度及失败保留测试通过；实际配置生成也已用推送记录验证；新的Isaac支撑与真实可见性待Spark运行。
可用`--native-report /绝对路径/report.json`指定基准记录。

结束后：

```bash
git add -- diagnostics/clean_scene_capture/
git commit -m "Add unmarked proxy scene initialization comparison"
git push origin main
```

## 下一轮：冻结两组无标记观测的原图/左右翻转模型对照

`frozen_image_probe.py`从最近完成且几何检查通过的clean_scene_capture中读取arm-only与arm-and-target-xy，
对每组分别查询no_flip和flip_x：两路相机一起水平翻转，状态与提示词保持不变。
默认每条件重复10次，共2组×2条件×10=40次推理。奇偶轮交替条件顺序；没有共享随机噪声种子，不宣称配对随机实验。
不启动仿真，不执行任何模型动作，不改变在线预处理或执行器参数。

发送的状态不是直接复用geometry_audit中的原始Isaac状态：

- 103.4mm传感器点沿当前手部局部轴修正-6.4mm，换为97mm TCP。
- 使用已通过多姿态验证的世界R/t转换TCP位置，手部姿态也乘同一个世界旋转。
- 通过注册四元数与原始手部四元数相乘，保留原始符号分支，不将手部强制规范成w>=0；再用既有LIBERO旋转向量编码。
- 双指状态按[q1,-q2]转换，保留当前实际张开位置。

立方体中心只用于注册世界中的局部趋近投影，不包含在模型输入字典中；输入只有8维state、两张224RGB、提示词。
完整动作原始值与假想1mm/0.25度限幅结果均保存，记录前5步、前10步、全chunk的投影与闭爪请求。
若返回超过10步，也验证全部未来动作；少于10步会停止，不默默截断或补零。
这些投影是固定观测方向上的数值诊断，不是实际行程或抓取成功率，不进行IK/旋转杠杆臂执行。

源manifest/result/geometry/setup/图像SHA均记录；逐像素确认flip_x副本只是原224RGB左右翻转，不另做resize或180度旋转。
每次实际发送float32状态、输入SHA、耗时、完整输出、按条件的均值/样本标准差/范围都会保存。
提示词默认沿用当前方块任务的`pick up the cube`，两组/两条件相同；黑碗任务的物体、桌面外观和语义差异仍存在。
两组结果分开统计，第二组仍有方块触边这一已知限制，不能把它与第一组混成同一个输入样本。

保持原有OpenPI服务在127.0.0.1:8000运行。在另一个Spark终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/.venv/bin/python \
  openpi_isaac_bridge/frozen_image_probe.py --repeats 10
```

使用模型环境中的现有客户端，不安装任何包。启动前先校验全部输入，随后检查端口可达；服务不可达会保存错误，不进入客户端无限重连。
可用`--source-manifest /绝对路径/run.json`指定输入；`--server-host`/`--server-port`选择现有服务。
可选`--policy-server-log /已有日志路径`复制128KiB以内日志前缀以辅助记录，但日志和握手metadata不能独立证明checkpoint身份。

输出 `diagnostics/openpi_image_ab/<UTC>/report.json`，含两组图像副本、原几何及相机setup、世界配准与输入来源副本。
异常保留已经完成的查询。actions_executed始终为0，policy_queries成功完成时为40。
200项本地CPU、源输入校验、符号/坐标转换、模型客户端模拟与完整CLI流程测试通过；实际b942542a4757来源的两组输入准备也已本地验证，真实模型推理待Spark运行。

结束后：

```bash
git add -- diagnostics/openpi_image_ab/
git commit -m "Add frozen original versus horizontal flip policy comparison"
git push origin main
```

## 下一轮：注册坐标与TCP补偿后的短动作回放

上一轮0683a3eefcc1完成40次冻结观测推理、没有执行动作。左右翻转使第二组的局部趋近投影改善，
但这只说明输出方向变化，不能证明实际运动或抓取成功。本轮先检查转换与短时跟踪，不进入闭环抓取。

`run_registered_replay_capture.py`读取最近完成的冻结A/B报告，分别在arm-only和arm-and-target-xy中，
选择flip_x条件下“前5步限幅趋近投影最接近本组平均值”的一条完整输出；并列时选较早repeat。
对0683a3eefcc1来源，选择repeat 2和repeat 9；选择依据是上一轮1mm/0.25度诊断值，本轮执行另用0.5mm/0.1度限制。
每组启动独立场景，使用源记录实际7关节角、已验证相机及相同方块XY。源报告、请求、SHA和所选完整输出均保存。
无需启动OpenPI服务，不进行新推理；图像翻转条件只决定读取哪组已有模型输出。

每个模型动作先在LIBERO世界轴中解码（平移乘0.05、旋转乘0.5），分别限制向量模长到0.5mm和0.1度，
再用配准旋转的转置变换到Isaac世界轴。增量不加世界平移；世界平移只用于初始化一致性检查。
以当前手部姿态计算TCP97当前点，并生成TCP97目标；目标手部姿态按世界轴旋转左乘：

```text
p97_target = p97_current + R_LI.T @ bounded_delta_L
q_hand_target = q_delta_I * q_hand_current
p107_target = p97_target + R(q_hand_target) @ [0, 0, 0.010]
```

然后将107mm IK目标变换到机器人根坐标，与实际IK107当前位姿求相对误差，再除以IK动作scale。
这里显式换算控制点，不能直接把103.4mm传感器点误差送入107mm IK。
先核对同一手部刚体、传感器/IK固定偏置与内部闭环误差；初始化与冻结源须在0.5mm/0.1度、关节0.001rad以内。

只执行前5步，每步50ms且夹爪强制张开；接着保持最后一个**绝对目标**10步，不将零动作重新锚定为当前位姿。
每组共0.75秒仿真动作/保持，预热另计。沿用当前执行器增益与10ms物理步长。
每控制周期记录手部/TCP97实际位姿、TCP97/IK107目标、实际关节角/速度、可用的关节目标、物理子步数和两路RGB。
这是50ms控制周期记录，没有新增逐物理子步采样器。

运行中检查单周期1.5mm/0.3度、总4mm/0.7度的实际运动范围，方块位移<=2mm、夹爪张开和关节软限位；
异常保留已执行记录并停止后续动作。最后目标误差<=0.25mm/0.1度单独记为final_target_reached；
status=completed只表示执行与检查流程完成，不自动代表目标收敛或抓取成功。

在Spark执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_registered_replay_capture.py
```

可用`--source-report /绝对路径/report.json`指定冻结A/B来源。输出到`diagnostics/registered_replay/<UTC>/`。
207项本地测试通过，包括独立旋转矩阵/TCP杠杆臂闭环、世界/根坐标往返、全chunk校验、两组调度和失败保留；
两组真实来源的5步转换也已离线验证。实际Isaac接口执行与动态结果待Spark运行。

完成后推送：

```bash
git add -- diagnostics/registered_replay/
git commit -m "Add short registered frozen action replay results"
git push origin main
```

## 下一轮：实时注册观测、一次新推理与短动作执行

57f68f80de69的两组冻结动作回放已在Spark执行：5动作+10保持均完成，最终位置误差约0.00384/0.02360mm。
这是仿真内部目标跟踪误差，不等于跨场景配准或实体机器人的精度；第二组仍存在首周期滞后和保持初期振荡。
本轮沿用同一场景、执行器、50ms控制周期、0.5mm/0.1度动作限幅和张爪条件，只更换动作来源为本次实时观测的新推理。

`run_online_registered_capture.py`从最近两组最终目标均通过的registered_replay/run.json读取场景配置，
核对前轮结果与世界配准SHA、初始角度和坐标参数一致性，然后分别启动两个独立Isaac进程。
每组完成35步张爪保持预热，保存本次无标记图像和geometry_audit，检查相机、物体支撑及初始化后：

1. 从本次几何记录构造LIBERO世界TCP97位置、手部旋转向量及[q1,-q2]状态，保留四元数符号分支。
2. 读取本次224RGB图像，核对SHA，逐像素确认两路输入都是原图的精确水平翻转；不读取旧实验图像作为新输入。
3. 发送一次`pick up the cube`请求，保存float32状态、图像/输入指纹、完整返回动作、推理耗时和握手metadata。
4. 验证整个动作chunk至少10步且全部有限值；只执行前5步，随后固定最后绝对目标保持10步。

每组只有一次查询，不重复采样挑选输出，也没有第二次重规划。推理期间不推进仿真，前后物理步计数必须相同。
动作执行复用已在Spark通过的世界反变换、TCP97到IK107补偿及运动/物体/关节检查。
模拟客户端、网络不可达、未来动作NaN、图像变化、仿真步数变化均有本地测试；任何失败保留已有报告并停止后续动作。
新模型输出无效时保留响应文件，没有执行动作；查询响应已收到与动作实际执行次数分别记录。

本轮仍是一次推理后的短开环动作段，没有验证多轮视觉反馈、接触、闭爪、抓取或抬升。
两组输入场景不同，第二组方块靠近腕部图像边缘这一限制仍然存在。
status=completed与final_target_reached继续分别记录，不将运行结束自动解释为目标达成。

保持原来的OpenPI服务在127.0.0.1:8000运行。如果服务已停止，在独立终端使用此前已验证的启动方式：

```bash
cd /home/test/Robot/openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

在另一个Spark终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_online_registered_capture.py
```

外层使用原有参考venv，内部Isaac通过`uv run --extra viser --extra openpi isaaclab -p`调用现有客户端。
可用`--source-manifest /绝对路径/run.json`指定前轮回放，`--server-host`/`--server-port`选择服务；提示词默认继承前轮冻结报告。
输出到`diagnostics/online_registered/<UTC>/`，每组包含online_registered_result.json、online_model_response.json、
online_replay_request.json、registered_replay_result.json、输入图像、15周期双相机图像和日志。
握手metadata不能独立证明checkpoint身份，推理墙钟耗时不作为真实机器人实时性保证。

213项本地测试通过；上一轮两组真实采集文件的新输入准备也已离线验证。新的一次真实模型查询与Isaac运动待Spark运行。

结束后推送：

```bash
git add -- diagnostics/online_registered/
git commit -m "Add live registered observation short motion results"
git push origin main
```

## 下一轮：同一场景内连续三轮重新观测与规划

610766962a8b包含三次独立重置运行，每次两组各查询一次，六段均完成并最终收敛。
arm-only最终位置误差约0.0037–0.0049mm，arm-and-target-xy约0.0115–0.0187mm。
后者仍有首步滞后，且方块靠近腕部画面边缘。本轮先只使用目标较居中的arm-only。

`run_replanning_capture.py`读取最近通过的online_registered/run.json，核对arm-only来源、结果与配准SHA，
使用相同初始化、相机、执行器增益和提示词，启动一个Isaac进程。在同一环境中执行：

```text
35步初始化张爪保持
观测1 → 推理1 → 执行前5步
观测2 → 推理2 → 执行前5步
观测3 → 推理3 → 执行前5步 → 保持最后绝对目标10步
```

三轮之间没有环境重置、额外保持或速度清零。共3次查询、15个动作周期和10个最终保持周期，
动作/保持合计1.25秒仿真时间，初始化和推理墙钟时间另计。
每轮继续保持夹爪张开，平移/旋转限幅为0.5mm/0.1度，沿用注册世界变换与TCP97到IK107补偿。
每个动作仍以当前实际位姿生成相对目标，没有额外累加历史未完成增量。

新增模式`three-round-replan`通过明确的online-registered输入启用，原单次推理与冻结回放默认行为不变。
底层回放仅新增显式0或10步保持选择：前两轮0步保持，最后一轮10步。
原单段4mm/0.7度及单周期检查继续生效；同时三轮共用最初参考点，逐周期检查总位移<=10mm、
总姿态变化<=2度、物体相对最初位置变化<=2mm，避免每轮重置运动检查基准。

后两轮从仿真当前缓冲重新读取双相机RGB及实际关节/手部/TCP位姿，重新生成geometry_audit和224左右翻转输入。
后续动态捕获不再拿已运动的关节角与原始初始化角度比较；初始化检查只在最开始做一次，之后验证当前几何闭环与总运动范围。
每两个观测之间必须恰好推进25个物理子步（5个50ms控制周期），两路相机帧号都必须增加。
逐轮记录输入指纹、状态位置变化和各图像是否逐像素变化；图像可能因像素量化而相同，不强制把图像变化作为通过条件。
`all_followup_inputs_changed`单独记录后两轮输入是否变化，不将新帧号与画面内容变化混为一谈。

每次推理期间仿真停步并保留实际速度，返回后继续推进仿真，查询前后步数必须相同。
这用于检查同步仿真的反馈链路，不保证实体机器人能够暂停等待模型。
上一轮出现过3.16秒推理离群值，本轮保留逐轮耗时，不预设其原因或承诺实时性。

保持原OpenPI服务在127.0.0.1:8000运行。在另一个Spark终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_replanning_capture.py
```

可用`--source-manifest /绝对路径/run.json`指定上一轮在线结果，`--server-host`/`--server-port`指定服务。
输出`diagnostics/registered_replanning/<UTC>/`，每次捕获目录内有round_01/02/03及总online_registered_result.json。
每轮保存本轮RGB、原始/水平翻转224输入、几何、输入状态与指纹、完整模型返回、执行请求、逐步运动记录与相机图像。
总报告区分前两轮动作段结束误差与最后保持后的收敛，不要求未保持的中间轮满足最终收敛阈值。
错误保留已完成的动作/保持计数和当前轮报告，不再进行下一轮查询。

218项本地测试通过，包括同一环境三轮调度、0/0/10保持、全程共用运动范围、输入更新、相机帧与步数检查、
第二轮非法未来动作/过期帧/推理期间推进物理时停止，以及外层单进程调度与失败保留。
上一轮真实arm-only采集已离线复现相同输入指纹；后两轮的实际Isaac动态捕获与模型反馈效果待Spark验证。
本轮仍不验证接触、闭爪、抓取或抬升。

完成后：

```bash
git add -- diagnostics/registered_replanning/
git commit -m "Add three-round registered replanning results"
git push origin main
```

## 抓取实验：OpenPI趋近/闭爪请求与程序抬升验证

1843fd4adb66已完成同一场景3轮反馈，输入确实更新，关节位置和速度在轮次之间连续。
本轮在同样的arm-only方块代理场景中尝试抓取，沿用相机、配准、提示词与执行器增益。
模型的原训练场景与当前代理场景差异继续存在；测试结果需要由实际物体运动判断。

`run_openpi_grasp_capture.py`使用最近通过的registered_replanning/run.json作为来源，核对源结果、世界配准SHA及初始化配置。
启用明确的`openpi-grasp`模式，默认最多200次新推理，每次验证整个至少10步的动作序列，只执行前5步。
趋近预算最多1000个50ms控制周期，即50秒仿真运动；初始化、推理、归档和验证动作时间另计。
保持既有0.5mm/0.1度向量限幅，不增加驱动增益。相机/状态每5步重新采集，推理期间同步仿真停步。

阶段与判据：

1. 趋近：OpenPI决定平移/旋转，夹爪保持张开。每100个动作周期至少缩短2mm物体距离；
   距离比初始增加超过2cm、方块在提起前移动超过2cm、掉落或控制采样估计速度超过0.5m/s会停止。
2. 模型闭爪请求：只有原始夹爪动作>0时才考虑闭爪。TCP横向距物体中心<=12mm、
   TCP比物体中心高-5到20mm、总距离<=15mm、手部局部Z轴朝下偏差<=20度时放行。
   未对准的请求只计数，继续张爪；没有模型闭爪请求时不自动强制抓取。
3. 闭爪检查：冻结当前位置/姿态并关闭夹爪60周期（3秒仿真）。检查双指仍有阻挡间隙、
   最近10周期稳定、左右对称、物体距TCP<=15mm以及位姿保持。空夹不进入抬升。
4. 验证抬升：通过闭爪检查后，由程序生成当前TCP向上5cm目标并保持闭爪，最多250周期到达。
5. 验证保持：保持40周期（2秒仿真）。物体必须始终比初始支撑高至少35mm，
   保持中相对TCP漂移<=10mm、相对旋转漂移<=10度，整个提起过程相对位置变化<=15mm，双指保持阻挡。

这是一项混合实验：OpenPI决定趋近运动和闭爪请求，仿真真值用于闭爪放行与成功检查，程序生成抬升/保持目标。
报告不将程序验证抬升计为模型自主完成，也不将几何阻挡候选计为测得双指接触力。
TCP97输入、世界坐标反变换和IK107目标补偿沿用已通过的实现；固定验证目标也通过同样0.5mm/0.1度中间目标推进。

全程额外检查单周期运动<=1.5mm/0.3度、相对最初TCP净位移<=35cm、手部姿态变化<=90度、
累计实际TCP路径<=85cm及关节软限位。物体速度是50ms采样位置差的估计，不是逐物理子步接触冲量检测。
任一停止条件保留已执行记录；status=completed只表示实验正常结束，抓取是否通过看success与outcome。
例如预算耗尽、空夹、抬升未带动物体或保持滑落都应为success=false。

保持原OpenPI服务在127.0.0.1:8000运行。在另一个Spark终端执行：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_openpi_grasp_capture.py --max-queries 200
```

可用`--source-manifest /绝对路径/run.json`指定三轮反馈来源，`--server-host`/`--server-port`指定现有服务。
需要重启模型服务时，沿用此前已验证的方式：

```bash
cd /home/test/Robot/openpi
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

输出`diagnostics/openpi_grasp/<UTC>/`。总run.json链接到捕获目录中的online_registered_result.json，
其中success/outcome/grasp_result记录结果，native_controls_executed记录实际env.step次数。
motion.jsonl逐周期保存尝试/结果事件，含关节位置速度、TCP、物体相对位姿、目标、实际夹爪动作及模型动作来源。
每20个周期和阶段变化时保存两路运动图像；每次查询的完整geometry、原始/策略图像、输入指纹、
原始返回和执行请求保存为queries/query_NNNN.zip，顶层查询记录含归档SHA。
ZIP用于控制长实验文件数量，输入图像仍以无损PNG完整保留。
失败或停止后也请推送整个输出目录，以便区分模型方向、闭爪门槛、动态跟踪与实际夹持问题。

228项本地测试通过，包括闭爪几何门槛、空夹不抬升、物体打飞/滑落/未随动时失败、模型未请求闭爪时不代替闭爪、
固定目标TCP补偿与限幅、完整未来动作验证、查询归档及非法输出保留、外层完成/成功分离。
最新推送来源的配置和配准也已离线核对。物理接触、夹持能力和真实模型长时运动仍待Spark执行验证。

实验结束后推送：

```bash
git add -- diagnostics/openpi_grasp/
git commit -m "Add gated OpenPI grasp experiment results"
git push origin main
```

## 修复抓取入口的空TCP探测日志

70b59de7282e完成140次真实推理和700步趋近动作，推理耗时约0.251–0.261秒。
服务端出现的`InvalidMessage: did not receive a valid HTTP request`及`stream ends after 0 bytes`，
与抓取外层和Isaac内部两处`socket.create_connection(...); close`端口检查一致：它们只建立TCP后立即断开，
WebSocket服务尝试读取HTTP升级请求时读到EOF。随后正常客户端仍可连接和推理。

新增`openpi_health.py`，抓取入口两处检查均改为完整`GET /healthz HTTP`请求，要求200及正文OK，保留3秒连接超时。
该路由来自OpenPI官方`websocket_policy_server._health_check`：
https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py
健康检查只确认服务HTTP路由，不能确认checkpoint身份或模型输出质量。
旧服务若未实现此路由会明确报HTTP状态，不能把任意端口可连接当作模型服务健康。

更新代码后可单独检查服务，无需启动Isaac或重新运行整段抓取：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/openpi_health.py --host 127.0.0.1 --port 8000
```

预期输出status=healthy、http_status=200。本次修复范围是抓取入口的健康检查。
232项本地测试通过，包括真实本地HTTP请求、错误状态/其他网页拒绝和断线失败保存；Spark现有服务的健康路由待实际调用。

本轮抓取没有成功，实际停止原因为`approach_progress_less_than_2mm_per100_controls`。
物体距离从246.4mm降至最近51.8mm后又增大到63.5mm；最后100步距离反而增加10.4mm。
303次模型闭爪请求全部因横向偏差不满足条件而被拦截，没有实际闭爪，也没有执行程序抬升。
最近点横向偏48.3mm、TCP高于方块中心18.6mm；最后横向偏63.0mm、TCP比中心低8.1mm。

单步位置跟踪误差中位0.0438mm、95分位0.3170mm、最大0.6092mm；
但当前逐步相对目标会丢弃前一步未完成量，700步的有符号误差会积累：
平移指令增量总和XYZ约[4.31,22.19,-246.28]mm，实际净位移约[-10.47,23.74,-248.85]mm，
X方向另少14.77mm。模型累计指令本身也没有保持对准方块，需分别分析模型方向与累计跟踪差异。
这组结果不能仅凭单步误差小就排除控制影响，也不能将失败归因于上述空握手日志。
