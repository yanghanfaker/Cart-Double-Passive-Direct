# Isaac：模型夹爪放行与动作幅度覆盖测试

## 目的

原生OSC固定动作回放显示，小平移限幅大幅改变轨迹，闭合延迟影响抓持；原生成功也不要求每50ms完全到位。
本轮回到既有Isaac方块代理场景，保持已确认的关节增益、夹爪驱动、初态、相机、TCP转换和输入格式。
分别改变夹爪指令是否放行、平移/旋转限幅，并增加相同Isaac归档动作的配对回放。
没有更换为OSC，不宣称与LIBERO碗场景等价。

## 在Spark运行

保持已验证OpenPI LIBERO服务在127.0.0.1:8000运行，保留先前diagnostics/execution_contact及既有几何/静态初态/场景来源。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/run_execution_alignment.py --suite policy_semantics
```

Isaac通过现有isaaclab -p启动链路加载；上面的LIBERO venv仅运行调度器，不直接导入Isaac Sim。
结果：diagnostics/execution_policy_semantics/<UTC>/。主终端显示工况，对应console.log记录完整模拟输出。

## 26个工况

| 分组 | 数量 | 条件 |
|---|---:|---|
| 程序抓取参考 | 2 | 首尾各一次既有h5、主动夹爪350/50程序抓取 |
| 在线模型动作 | 16 | 下列4种幅度 × 强制张开/放行模型夹爪 × 两重复，第二轮反向顺序 |
| 相同动作离线回放 | 8 | 第一条0.5mm/0.1度张开在线轨迹的实际前最多100步 × 同样4种幅度和2种夹爪 |

四种幅度组合：

| 平移向量模长上限 | 旋转向量模长上限 |
|---:|---:|
| 0.5mm | 0.1度 |
| 2mm | 0.1度 |
| 5mm | 0.1度 |
| 5mm | 0.5度 |

各在线工况最多300个50ms控制周期，每次执行前5步后用新图像/状态查询；推理期间物理不推进。
5mm仍小于原生每轴50mm的缩放范围，只是扩大动作幅度的诊断档位，并非取消限制或最终参数。
J1/J3=1687.5/50，J2/J4=5400/120，J5/J6/J7=900/20；继续检查运行时读回。
夹爪主动350/50、现有被动/mimic设置、每10ms子步0.1m/s目标斜坡均保持。

## 与旧接近流程的明确区别

在线组不做几何门控、程序细调、程序闭合或程序抬升。
模型组将正第7维转换为Isaac闭合-1，负值转换为张开+1，零保持前一命令；实际闭合仍按既有斜坡执行。
强制张开组逐步保存原始第7维，但实际发送张开。
两组使用相同动作缩放、坐标/TCP变换与周期。

模型组使用显式policy-model阶段，因此不会因正常闭合触发“approach gripper not open”，也不会因正常搬运触发“抬升前物体位移2cm”。
只有声明的本轮模型工况可使用此阶段；旧hybrid、contact、online_caps的逻辑保持。
强制张开组使用policy-open，保留实际手指至少30mm张开的检查和接近时物体位移2cm的检查。

继续保留关节软限位、既有驱动力/速度上限、周期IK修正及实际位移边界、整体0.35m位移/0.85m路径/90度旋转边界、
物体速度0.5m/s、跌落检查和10ms轨迹审计。
局部边界按已有规则随声明限幅变化：至少1.5mm/0.3度，或声明限幅的1.5倍。
另保留TCP低于初始物体原点1cm的停止条件；尚无明显抬升时，距离增加2cm或每100周期进展不足2mm会停止。
保护触发需保留原因、最后物理步和已执行前缀，不强行运行到预算结束。
速度/力矩缓冲不是接触力；碰撞、材质和物理步长不改动。

## 结果判定

模型动作连续40个控制周期满足物体高于初态至少35mm、物体相对TCP漂移不超过10mm、相对旋转漂移不超过10度、
两手指实际位置均在2–39mm之间，记model_lift_hold_observed。
该判定只读仿真真值，不将真值目标输入策略，不代替模型生成任何保持动作。
这是约2秒的模型驱动抬升/相对稳定证据，不是实测接触力、完整放置任务或原生LIBERO成功率。
峰值物体高度、最近距离、原始/实际夹爪请求及指位也单独记录，避免将所有未完成都记作同一种失败。

首个程序参考必须实际成功才启动依赖组；末尾参考再次检查链路。
根completed只说明执行/记录完整，不要求在线模型组都抓取成功；保护触发、执行失败或identity不复现会使根needs-inspection。

## 离线配对

SOURCE为semantics-live-open-0.5mm-0.1deg-r1。必须正常完成记录，并至少实际执行5个控制周期。
取其真实前min(100,实际步数)个动作，验证响应ZIP、请求、动作顺序、查询期间物理步不变和源结果摘要。
复制source_result、source_row、选用的完整查询ZIP和request到本轮replay_inputs；不补零、不循环、不额外等待。
回放零新推理，原始动作保持相同；它不是对干预后新观测重新推理的闭环实验。

先执行0.5mm/0.1度/张开的identity，检查每一步前后TCP/物体/手指差<=0.1mm，关节差<=1mrad，
关节速度差<=0.001rad/s，手部/物体角度差<=0.05度。失败则跳过其他配对回放。
在线两重复没有相同服务端随机种子保证，不能当作严格配对；离线同动作组用于隔离执行干预。

## 记录与分析

- run.json：工况、来源、依赖、代码摘要、实际推理次数和各组summary。
- capture/.../online_registered_result.json：settings、逐动作trace、policy_semantics决策与grasp_result。
- 原始目标：target.decoded_delta_L_m/decoded_rotation_L_rad；限幅后目标：bounded_delta_I_m及TCP97/IK107目标。
- motion.jsonl：实际末端响应与周期末TCP/姿态目标误差。trace中的target.gripper_applied与实际下发值一致。
- physics_joint_trace.jsonl.gz：每10ms七关节位置目标/实际位置/速度、雅可比、驱动力缓冲、手指和物体状态。
- queries/query_*.zip：在线实际图像、状态、提示词、完整响应和执行前缀；回放使用replay_inputs来源，无新查询。

先比较同幅度下夹爪放行是否改变抓持/反馈，再比较同夹爪下幅度改变目标与实际响应的程度。
5mm/0.5度组区分旋转幅度影响；同动作回放先验证identity，再比较共同执行前缀。
不能仅凭更小绝对目标误差判断控制更好：目标本身被压小也会让误差变小。
若仍主要受额外限幅影响，下一轮再设计受控扩幅；若下发目标合理但关节明显落后，再检查IK/驱动动态。

## 推送结果

```bash
git add -- diagnostics/execution_policy_semantics/
git commit -m "Add Isaac policy gripper and action amplitude comparison results"
git push origin main
```
