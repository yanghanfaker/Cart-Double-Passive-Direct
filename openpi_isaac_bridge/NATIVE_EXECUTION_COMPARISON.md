# 原生 OSC：动作限幅与夹爪延迟配对回放

本轮复用上一轮 native_gripper_comparison 的6条成功在线源轨迹，隔离桥接层改变动作幅度和夹爪时序的影响。
不重新查询模型，不修改原生 OSC、增益、物理步长、场景和碰撞；不是 Isaac 实验，也不是官方完整评测。
先确定动作干预本身的影响，再决定 Isaac 上应对照哪些控制差异，避免同时更换控制器、增益、限幅和夹爪规则。

## 运行

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/native_execution_comparison.py
```

无需启动模型服务器；保留上一轮 diagnostics/native_gripper_comparison 的完整文件。
默认选择最新 completed 来源；可用 --source-run /绝对路径/run.json 指定。
仍使用现有 LIBERO venv、EGL 和受限状态加载，不重新安装依赖。
结果位于 diagnostics/native_execution_comparison/<UTC>/。每回合独立进程，最长15分钟；终端打印工况，worker.log保留原始输出。

## 60个工况

任务0、2 × 初态0、1、2 × 下列10种条件。每个初态先做identity，物理状态与输入摘要均复现才执行依赖组。

| 条件 | 平移向量模长限制 | 旋转向量模长限制 | 夹爪 |
|---|---|---|---|
| identity | 仅原生限制 | 仅原生限制 | 源原值 |
| open | 仅原生限制 | 仅原生限制 | 始终-1张开 |
| delay5 | 仅原生限制 | 仅原生限制 | 每段连续闭合请求先阻止5周期，250ms |
| delay10 | 仅原生限制 | 仅原生限制 | 每段连续闭合请求先阻止10周期，500ms |
| translation0p5 | 0.5mm | 仅原生限制 | 源原值 |
| translation1 | 1mm | 仅原生限制 | 源原值 |
| translation2 | 2mm | 仅原生限制 | 源原值 |
| rotation0p1 | 仅原生限制 | 0.1度 | 源原值 |
| both0p5 | 0.5mm | 0.1度 | 源原值 |
| both0p5-delay5 | 0.5mm | 0.1度 | 阻止连续闭合前5周期 |

限幅按当前Isaac桥接的顺序计算：每轴clip到[-1,1]，乘平移0.05m/旋转0.5rad，再限制向量模长，最后还原为原生动作域。
没有该项限幅的条件保留原始对应维度，由原生OSC正常clip，不提前改变原始动作。
不引入额外累计目标，不改变50ms策略周期或原生2ms物理步长。

延迟是闭合请求抑制：遇到第7维<=0立即重置计数，短于延迟窗口的闭合段会全部被阻止。
窗口内发送-1张开；不是将所有夹爪动作整体后移，也不是延迟启动后补执行。
这是时序敏感性诊断，不是Isaac几何门控的等价实现。源第7维的幅值在放行后保持原值。

## 来源与记录

- 核对上一轮根状态、6个物理复现配对、每条源worker/episode/推理ZIP摘要、成功状态和实际执行前缀。
- 保存source_run.json及6条完整源响应归档、初态、场景、BDDL和控制器源码；状态文件继续使用固定清单验证。
- 每组只执行对应源轨迹的实际控制数；保持前5步请求归档响应。成功或done提前结束，绝不补零、循环、延长等待。
- raw_action与实际action逐步并存，核验唯一声明的干预，回放new_policy_queries必须为0。
- 每次请求仍归档当前图像/状态和取出的历史响应，明确query_origin=archive；干预组观测摘要变化是预期现象。
- 各组配置必须与源的提示词、初态、BDDL、物体、OSC参数和20Hz周期一致。
- 保存原生夹爪formatter状态、actuator ctrl和每周期末的OSC目标、末端缓冲、速度、关节状态及力矩命令。

controller_tracking读取最后一次OSC求值的已有缓冲，不调用forward/update；它可能早于最后一个物理积分步。
其中position_error_m使用同次缓冲内goal_pos与ee_pos，不能直接与末帧图像当作完全同时。
torques是控制器命令，不是实测接触力。诊断不可用明确记录，工况标记failed，不伪造0误差。

## 解释方法

先看identity的全状态最大差<=1e-7及全部查询输入摘要一致；否则同初态其他组跳过。
每组逐步重新计算声明干预，核对50ms步进，保存共同前缀的最大EEF位置/姿态差、夹爪改变次数、限幅次数和OSC跟踪误差。
提前终止只比较实际共同前缀，记录源长度和实际长度。

1. identity与open：复核只改变夹爪的直接物理影响。
2. identity与delay5/delay10：看闭合时序变化，而不是始终张开的极端条件。
3. identity与三档translation：隔离平移幅度压缩。
4. identity与rotation0p1：隔离旋转幅度压缩。
5. 单项与both0p5、both0p5-delay5：检查组合影响。
6. 对应OSC跟踪误差和末端轨迹：区分目标已经缩小与目标仍未跟上；这里不能推断Isaac关节跟踪一定正常。

所有干预组使用源有限时域内的固定动作，后续动作并未根据干预后状态重新规划。
因此“未成功”只能解释为该归档动作序列在此干预和时域下未完成，不能当作在线策略成功率，也不能据此宣称限幅或延迟单独导致全部Isaac故障。
本轮不解决原生碗/Isaac方块域差异、TCP配准或OSC/IK动力学等价性。
completed只表示工况和记录核验完成，不要求所有干预组抓取成功。

## 推送

```bash
git add -- diagnostics/native_execution_comparison/
git commit -m "Add native action cap and gripper delay replay results"
git push origin main
```
