# 小车驱动的二级倒立摆（Isaac Lab）

任务 ID：`Isaac-Cart-Double-Passive-Direct`。

策略只控制小车水平推力，两个摆关节的驱动刚度、驱动阻尼和目标力矩均为零。观测为小车位置和速度、两根杆相对直立方向的绝对角度和角速度，共 6 维。动作只有 1 维，限制为 ±100 N。

本任务从接近直立的随机初态学习平衡，不包含从下垂位置起摆。

## 运行前提

- 面向具有 `isaaclab_tasks/core/cartpole` 目录和统一 `isaaclab train/play` 命令的 Isaac Lab 3 开发版本。
- Isaac Lab 已能使用 `physics=isaacsim_physx` 运行官方 Cartpole 任务。
- RSL-RL 已安装。
- 本代码仅完成语法检查，尚未在 DGX Spark 上完成运行或收敛验证。开发分支 API 变化可能需要适配。

## 安装

在本仓库目录执行，参数为已有的 Isaac Lab 根目录：

```bash
python3 install_passive_double.py /home/test/Robot/IsaacLab
```

脚本新增任务文件，在官方 Cartpole 包的 `__init__.py` 末尾增加导入。修改前自动生成带时间戳的备份。不会覆盖官方 Cartpole 环境和训练配置。

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

## 更新

在本仓库执行 `git pull` 后，重新运行安装脚本，然后重新启动训练或回放。不要在训练过程中更换任务代码。

## 验收要点

- 启动日志显示 1 维动作和 6 维观测。
- 两根摆杆关节自由运动，不受位置或速度驱动锁定。
- `_apply_action()` 给所有关节写零目标力矩，仅覆盖小车滑动关节为非零推力。
- `Metrics/survival_rate` 表示回合撑到超时的比例，不是泛化成功率；仍需用不同初始状态、多次回放评估。
