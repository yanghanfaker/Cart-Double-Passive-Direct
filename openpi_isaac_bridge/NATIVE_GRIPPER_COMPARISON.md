# 原生 LIBERO 夹爪干预覆盖测试

## 目的与范围

官方客户端基线已在 Spark 跑出10任务9成功1失败，状态加载兼容通过。
本轮研究当前 Isaac 接近阶段强制张开是否改变原生执行轨迹或后续模型输出。
这是带逐步记录和干预的实验，不是未修改的官方评测；不修改 Isaac 场景、增益、限幅、碰撞或现有模型服务。
强制张开组本来就缺少正常抓持能力，不能仅用成功率降低作为模型失效的证据。

## 矩阵：24个独立回合

使用libero_spatial任务0（黑碗在盘子和小碗之间）、任务2（黑碗在桌面中心），各使用官方初态0、1、2。
每个任务/初态的四组如下：

| 模式 | 动作来源 | 夹爪 | 控制预算 | 主要作用 |
|---|---|---|---|---|
| model-live | 当前服务在线推理 | 模型原始第7维 | 最多220步，成功/终止提前结束 | 记录新的正常源轨迹 |
| model-replay | 同一源轨迹的完整原始响应 | 模型原始第7维 | 与源实际控制数完全相同 | 验证回放能复现源物理状态 |
| open-replay | 与上组逐值相同的原始响应 | 只将第7维替换为-1张开 | 与源实际控制数完全相同 | 隔离固定动作下夹爪干预的影响 |
| open-live | 每5步用新图像/状态重新推理 | 只将第7维替换为-1张开 | 最多220步 | 检查改变反馈后模型接近路径的变化 |

每个回合新建进程和环境，固定客户端种子7，加载相同任务/初态，先执行10个官方dummy wait动作。
保留原生20Hz控制、渲染/图像处理、提示词、动作缩放和控制器。
原生LIBERO动作+1闭合、-1张开；禁止修改前6维动作，无新增平移/旋转限幅。
所有模式执行前5步再查询/请求下一段。回放零新推理，不补零、不延长、不循环源动作。
第二个初态改变三种比较组的执行顺序，源在线组仍必须先运行。
在线源与在线张开组没有同服务端随机种子保证，不能把两次新模型响应当成严格配对样本。

## 运行

保留已验证的pi05_libero服务在127.0.0.1:8000，保留此前diagnostics/official_libero_baseline完整结果。
默认从最近一个完成10回合的官方基线读取来源，并核对官方源码、非空文件摘要和状态快照。
只有终局汇总后的EGL清理异常可留作背景记录；回合内Traceback/Caught exception会阻止用作来源。
使用已有LIBERO客户端venv，不重装torch、MuJoCo、robosuite或模型：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
/home/test/Robot/openpi/examples/libero/.venv/bin/python \
  openpi_isaac_bridge/native_gripper_comparison.py
```

结果：diagnostics/native_gripper_comparison/<UTC>/。
主终端逐个打印case名称；对应case/worker.log可查看查询步进。
每回合最长15分钟超时；中断时只终止该回合进程组，不停止独立模型服务。
来源在线组若运行失败，该初态的依赖比较组跳过，其他初态继续。

## 记录与判定

- 每个case保存case.json、report.json、episode/report.json、queries.zip、双相机preview.gif、初态npy、任务BDDL、场景XML和控制器/夹爪源码。
- queries.zip保存每次实际输入的两路图像、状态npy、提示词摘要和完整原始模型响应；在线查询期间仿真时间必须不变。
- episode.trace逐步保存raw_action、实际action、模型是否要求闭合、实际是否下发闭合、EEF/关节/手指/所有物体观测及原生夹爪归一化命令和actuator ctrl。
- actuator ctrl是执行器输入，不是测得接触力；诊断无法读取会明确标记unavailable。
- policy_queries保留历史API的infer调用计数；new_policy_queries明确区分在线真实查询与离线回放请求，回放必须为0。
- 距离仅为EEF到指定碗刚体原点，不是指尖到实际抓取面的距离，不套用Isaac代理方块的15mm闭合门槛。

根run.json的pairs必须先满足：

1. 初态文件/场景/控制器/目标对象配置相同。
2. 正常回放与源轨迹的EEF、关节、手指和物体状态最大绝对差<=1e-7。
3. 张开回放在首次阻止模型闭合之前也与正常回放一致。
4. 两种回放原始动作逐值相同，强制张开仅改变第7维。

再看首次被阻止闭合、实际指位、碗位移、EEF轨迹偏移和在线模型方向变化。
normal_replay_all_query_inputs_equal另行报告图像/状态摘要是否全部相同；图像差异不被隐藏。
若正常回放不能复现，不能将两组差异单独归因于夹爪；根状态为needs-inspection。
有模型闭合请求才说明施加了所研究的干预，intervention_observed=false的配对不能支持闭合影响结论。
completed表示记录/回放校验完成，不要求强制张开组成功，也不是24回合全部抓取成功。
这一轮没有模拟当前Isaac的几何到位后程序接管，只隔离接近阶段强制张开的影响。

## 下一步如何解释

- 正常源与正常回放均成功，而固定动作张开回放出现路径偏移：支持存在夹爪/接触/物体运动对轨迹的直接影响。
- 固定动作EEF轨迹差异很小，但在线张开组的模型方向显著变化：反馈通道值得优先检查；仍需考虑模型采样差异。
- 在线张开仍能接近目标，而Isaac同类行为不能接近：继续核对场景、物体、观测映射、TCP和控制器动态，不能由此直接锁定某一项。
- 原生和Isaac的物体/接触面并不相同，不能直接把原生碗体原点距离当作代理方块闭合门槛。

## 推送

```bash
git add -- diagnostics/native_gripper_comparison/
git commit -m "Add native gripper intervention and matched replay comparison results"
git push origin main
```
