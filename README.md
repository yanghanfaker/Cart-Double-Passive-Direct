# 小车驱动的二级倒立摆（Isaac Lab）

| 任务 ID | 初态与目标 |
| --- | --- |
| `Isaac-Cart-Double-Passive-SwingUp-Direct` | 两杆自然下垂，从下向上起摆并保持倒立（8 维观测） |
| `Isaac-Cart-Double-Passive-Direct` | 从完全向上直立开始，保持平衡（6 维观测） |

以下首先说明原有平衡任务；自然下垂起摆请使用后面的“从自然下垂起摆”命令。

策略只控制小车水平推力，两个摆关节的驱动刚度、驱动阻尼和目标力矩均为零。观测为小车位置和速度、两根杆相对直立方向的绝对角度和角速度，共 6 维。动作只有 1 维，限制为 ±100 N。

默认每回合从完全竖直向上的姿态开始：两个摆关节角度为 0，小车位于轨道中心，全部关节速度为 0。不包含从下垂位置起摆。这里规定的是初始化/重置时刻的状态；之后由物理仿真和策略动作决定运动。

如需恢复小角度随机初态，可在训练或回放命令中加入 `env.initial_angle=0.03`，两个相对关节角分别在 ±0.03 rad 内采样。完全直立且静止时系统处于理想平衡点；评估恢复能力时可启用下方的周期推力测试。

## 运行前提

- 面向具有 `isaaclab_tasks/core/cartpole` 目录和统一 `isaaclab train/play` 命令的 Isaac Lab 3 开发版本。
- Isaac Lab 已能使用 `physics=isaacsim_physx` 运行官方 Cartpole 任务。
- RSL-RL 已安装。
- 原平衡与起摆任务已由用户在 DGX Spark 跑通训练。新增随机推力通过调度与安装测试，尚待 Spark 回放验证；开发分支 API 变化可能需要适配。

## 安装

在本仓库目录执行，参数为已有的 Isaac Lab 根目录：

```bash
python3 install_passive_double.py /home/test/Robot/IsaacLab
```

脚本新增任务文件和 `push_disturbance.py` 辅助模块，在官方 Cartpole 包的 `__init__.py` 末尾增加导入。修改前自动生成带时间戳的备份。不会覆盖官方 Cartpole 环境和训练配置。

## 短训练检查

```bash
cd /home/test/Robot/IsaacLab
uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-Direct \
  --num_envs 64 --max_iterations 10 \
  physics=isaacsim_physx
```

## 训练

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-Direct \
  --num_envs 1024 --max_iterations 1500 \
  physics=isaacsim_physx
```

1500 轮为初始实验预算，不保证收敛。日志目录为 `logs/rsl_rl/passive_double_cartpole/`。

## 浏览器测试

```bash
uv run --extra viser isaaclab play --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-Direct \
  --checkpoint latest --num_envs 4 --viz viser \
  physics=isaacsim_physx
```

在 Windows 浏览器访问 Spark 上 Viser 日志打印的地址（默认端口 8080）。如果使用 SSH 隧道，在 Windows 执行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:127.0.0.1:8080 test@SPARK_IP
```

将 `SPARK_IP` 替换为实际地址，然后打开 `http://127.0.0.1:18080`。

一级倒立摆和官方多智能体双摆的模型不适用于此任务，需要重新训练。

## 从自然下垂起摆

任务：`Isaac-Cart-Double-Passive-SwingUp-Direct`。

- 重力为 `(0, 0, -9.81)` m/s²。第一摆关节为 π rad，第二个相对关节为 0；两个物理杆的绝对姿态均为竖直向下。
- 小车在导轨中心，初速度为零。导轨根坐标高度改为 4 m，给下垂和旋转留出地面间隙；需要在实际资产上观察确认间隙。
- 只有小车电机驱动，摆关节不锁定，不主动施力。
- 使用 8 维观测：小车位置/速度、两个绝对角的 sin/cos、两个绝对角速度，避免角度绕过 ±π 时观测跳变。
- 允许杆大角度旋转，不再因为倾斜而结束；小车越界或状态非有限时失败，20 秒超时重置。
- 奖励鼓励两杆上升、直立并减速。`Metrics/swingup_success_rate` 表示该回合曾连续满足直立、低速和小车靠近中心条件至少 1 秒的比例。`survival_rate` 只表示未越界撑到超时，不代表起摆成功。

先用零控制检查下垂姿态（不是训练）：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab zero_agent \
  --task Isaac-Cart-Double-Passive-SwingUp-Direct \
  --num_envs 1 --viz viser physics=isaacsim_physx
```

停止零控制测试，先做 10 轮训练检查：

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-SwingUp-Direct \
  --num_envs 64 --max_iterations 10 physics=isaacsim_physx
```

然后开始训练：

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-SwingUp-Direct \
  --num_envs 1024 --max_iterations 5000 physics=isaacsim_physx
```

5000 轮只是初始实验预算，尚未在 DGX Spark 验证起摆收敛，可能需要调整奖励、探索、轨道范围或训练预算。新日志目录是 `logs/rsl_rl/passive_double_swingup/`。

测试起摆模型：

```bash
uv run --extra viser isaaclab play --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-SwingUp-Direct \
  --checkpoint latest --num_envs 4 --viz viser physics=isaacsim_physx
```

请勿加载原来的 6 维平衡模型；新的 8 维网络必须重新训练。周期扰动力参数也可用于这个任务，但初次训练先保持 `env.push_force=0.0`。

若 Viser 报 `add_line_segments() ... unexpected keyword argument 'thickness'`，在 Isaac Lab 根目录执行 `uv add --optional viser "viser>=1.1.0"` 后重启回放。后续回放保留 `--extra viser`。若网页客户端版本不一致，关闭旧标签并强制刷新或用无痕窗口打开。

## 周期推力抗扰测试

扰动默认关闭，可在加载已有模型时启用，无需重新训练：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab play --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-Direct \
  --checkpoint latest --num_envs 4 --viz viser \
  physics=isaacsim_physx \
  env.push_force=5.0 env.push_env_id=0
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `env.push_force` | `0.0` | 附加力，单位 N；正负表示沿导轨的相反方向，0 关闭 |
| `env.push_env_id` | `0` | 只扰动第 0 个环境；`-1` 扰动全部环境 |
| `env.push_start_s` | `2.0` | 每回合第一次扰动开始的仿真时间 |
| `env.push_duration_s` | `0.1` | 单次推力持续的仿真时间 |
| `env.push_interval_s` | `3.0` | 推力开始时间之间的间隔 |

默认时间设置下，在第 2、5、8 秒附近推第 0 个环境，其余环境作为对照。时间按控制步取整，短于一个控制步的脉冲至少持续一步；持续时间取整后必须小于周期。每次环境重置后重新计时，若回合在 2 秒前结束，就不会受到该次推力。

实现把扰动力加在小车滑动关节的目标力上，在理想直线导轨模型中等效于沿轨道推小车。不是对摆杆任意位置施力，也不会在网页自动增加施力按钮。两个摆关节目标力矩仍为零。实际总推力仍受求解器/执行器上限影响。建议从 2 N、5 N、10 N 逐步测试；要做可复现实验，使用指定 checkpoint 路径替代 `latest`。

## 外部随机推力测试

加载已经训练的起摆模型，无需重新训练：

```bash
cd /home/test/Robot/IsaacLab
uv run --extra viser isaaclab play --rl_library rsl_rl \
  --task Isaac-Cart-Double-Passive-SwingUp-Direct \
  --checkpoint latest --num_envs 4 --viz viser \
  physics=isaacsim_physx \
  env.push_mode=random env.push_env_id=0 \
  env.push_start_s=5.0 env.push_random_seed=42 env.push_log=True
```

此命令只对第 0 个环境沿导轨施加随机方向的推力：幅值 2–5 N，持续时间 0.05–0.15 秒；每次结束后随机等待 2–4 秒。每回合重置后先留出 5 秒起摆，再等待 2–4 秒，因此第一次推力约在第 7–9 秒开始。回合提前结束时不会等到该次推力。其他三个环境不施加推力，可作定性对照。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `env.push_mode` | `periodic` | `random` 启用随机脉冲，`periodic` 使用原周期模式 |
| `env.random_push_force_min` | `2.0` | 随机力绝对值下限，N，须大于 0 |
| `env.random_push_force_max` | `5.0` | 随机力绝对值上限，N；符号独立随机选择 |
| `env.random_push_duration_min_s` | `0.05` | 持续时间下限 |
| `env.random_push_duration_max_s` | `0.15` | 持续时间上限 |
| `env.random_push_wait_min_s` | `2.0` | 脉冲结束至下次开始的等待时间下限 |
| `env.random_push_wait_max_s` | `4.0` | 等待时间上限 |
| `env.push_random_seed` | `42` | 扰动力专用种子，不修改策略或环境 RNG |
| `env.push_log` | `False` | 打印每次力开始/结束的环境 ID、总仿真时间、力和持续时间 |

随机模式不使用 `push_force`、`push_duration_s`、`push_interval_s` 来采样。要关闭扰动，使用 `env.push_mode=periodic env.push_force=0.0`。`push_env_id=-1` 表示给全部环境独立施力。随机调度使用每环境独立的 RNG，时间向上取整至控制步；相同种子、相同重置时序下可以复现。环境受力后重置时，旧推力立即清零，重新等待起摆宽限期。

输出示例：`[Random push] env=0 sim_time=7.400s force=-3.200N duration=0.100s`。`force=0` 表示该次推力结束。总仿真时间不是墙上时间，也不一定是当前回合时间。

该实现是加在小车滑动关节上的等效外部水平力，受求解器总力上限影响，不对摆杆任意位置施力。网页没有额外施力按钮。测试时固定 checkpoint，分别比较无扰动/2–5 N/5–10 N，多跑不同种子。`swingup_success_rate` 只要求回合内曾成功直立过，不能单独代表受力后的恢复能力；需观察受力后是否恢复、是否越界，以及连续直立时间。

仅在有需要时用所有环境随机受力训练；本实现的 Python 调度适合少量环境的测试，大规模训练可能增加开销。

调度器和安装脚本的独立检查（不需要安装 Isaac Lab）：

```bash
python3 -m unittest -v test_push_disturbance.py
```

## 更新安装

在本仓库执行 `git pull` 后，重新运行安装脚本，然后重新启动训练或回放。不要在训练过程中更换任务代码。

## 验收要点

- 启动日志显示 1 维动作和 6 维观测。
- 两根摆杆关节自由运动，不受位置或速度驱动锁定。
- `_apply_action()` 给所有关节写零目标力矩，仅覆盖小车滑动关节为非零推力。
- `Metrics/survival_rate` 表示回合撑到超时的比例，不是泛化成功率；仍需用不同初始状态、多次回放评估。
