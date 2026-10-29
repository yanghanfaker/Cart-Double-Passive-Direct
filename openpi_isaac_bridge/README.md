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
