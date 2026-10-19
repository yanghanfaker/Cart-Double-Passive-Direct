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
uv run isaaclab play --rl_library rsl_rl \
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

## 更新

在本仓库执行 `git pull` 后，重新运行安装脚本，然后重新启动训练或回放。不要在训练过程中更换任务代码。

## 验收要点

- 启动日志显示 1 维动作和 6 维观测。
- 两根摆杆关节自由运动，不受位置或速度驱动锁定。
- `_apply_action()` 给所有关节写零目标力矩，仅覆盖小车滑动关节为非零推力。
- `Metrics/survival_rate` 表示回合撑到超时的比例，不是泛化成功率；仍需用不同初始状态、多次回放评估。
