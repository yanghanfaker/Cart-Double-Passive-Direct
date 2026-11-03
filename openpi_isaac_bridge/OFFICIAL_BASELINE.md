# 官方抓取基线与 Spark ARM64 的执行边界

## 当前决定

用户只有 DGX Spark ARM64。先运行 OpenPI 官方 LIBERO 未修改客户端，验证当前 pi05_libero 在其原生场景/控制器中的表现。
这不是 Isaac Lab-Arena 已跑通，也不是 Arena 的 ARM 移植。现有 Isaac 桥接代码、限幅、增益、模型和场景均不修改。
此前的 native_libero_rollout.py 是带记录的自定义原生测试；本轮增加未修改的官方入口核对，避免自定义 rollout 本身成为混淆因素。

## 为什么暂不安装 Arena

- 固定 Arena 官方版本：[3b0f10ce2df6c25649afbbf5aed0b1cd27d5cd66](https://github.com/isaac-sim/IsaacLab-Arena/tree/3b0f10ce2df6c25649afbbf5aed0b1cd27d5cd66)。
- 该版本 pyproject.toml 的 tool.uv.environments 明确只支持 Linux x86_64。
- 其 Dockerfile 默认 Isaac Sim 6.2.0；网上安装文档曾显示6.0.1，不能混用滚动文档的版本数字。
- DROID 官方示例使用 Robotiq 夹爪、7维绝对关节位置+1维夹爪动作，不是当前 Panda 双指夹爪的 LIBERO 末端增量接口。
- Arena 配套模型 pi05_droid_jointpos_polaris，检查点 gs://openpi-assets-simeval/pi05_droid_jointpos，执行前15步；不能接当前 pi05_libero 服务。
- official_arena_manifest.json 固定15个相关源码文件的 SHA-256/Git blob SHA-1及配套 OpenPI 提交。

可在 Spark 记录平台检查；blocked-platform 是预期结果，不会安装软件或启动模型：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
python3 openpi_isaac_bridge/official_arena_preflight.py
```

不要删除官方平台约束、使用CPU架构模拟或覆盖现有 Isaac Lab 安装来把 ARM 移植标成官方基线。
以后有兼容 x86_64 NVIDIA GPU 主机时，再在独立目录固定该 Arena 提交，按该提交安装依赖、运行原 YAML，记录三回合视频和 arena_experiment_result.json。

## Spark 原生 OpenPI/LIBERO 基线

### 运行前提

保留之前已能推理的 OpenPI 服务在127.0.0.1:8000，确认加载的是 pi05_libero 和匹配的归一化数据。
不要换成 DROID 模型，不要直接重装 JAX/torch，不要覆盖之前解决 ARM bf16 编译问题的运行环境。
客户端使用现有 /home/test/Robot/openpi/examples/libero/.venv/bin/python；脚本保留这个路径，不 resolve 到系统 Python。
官方入口比之前带记录的原生脚本多用到 imageio 视频依赖。先做一次无仿真、无服务连接的完整检查：

```bash
python3 openpi_isaac_bridge/run_official_libero_baseline.py --check-only
```

run.json 的 runtime.failures 一次列出全部导入错误和 FFmpeg 编码器问题。已确认本次缺 imageio，可只补官方版本的视频依赖：

```bash
uv pip install --python /home/test/Robot/openpi/examples/libero/.venv/bin/python \
  "imageio==2.35.1" "imageio-ffmpeg==0.5.1"
```

若还报告 tyro/tqdm 缺失，对应官方版本为 tyro==0.9.2、tqdm==4.67.1。按实际缺失项补装。
若 imageio_ffmpeg 可导入但 FFmpeg 不可执行，先检查系统 ffmpeg -version；ARM64 不能假定包内携带可用的编码器。
有系统编码器时可 export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)" 后重查；系统未安装时需先安装适合该系统的 FFmpeg。
不要为补视频工具执行整份旧 requirements 的 uv pip sync，它还固定了旧 NumPy、torch/CUDA版本，会影响现有 ARM 环境。
视频依赖版本来源为该 OpenPI 提交的 examples/libero/requirements.txt。

父脚本只依赖标准库，可以用 Spark 的 python3 运行；子进程使用上述专用 venv。
需要当前网络能下载固定的 upstream main.py；下载字节必须匹配清单 SHA-256，否则拒绝执行。

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
git pull --ff-only
python3 openpi_isaac_bridge/run_official_libero_baseline.py
```

本轮固定：

| 项目 | 配置 |
|---|---|
| 官方入口 | OpenPI c23745b5ad24e98f66967ea795a07b2588ed6c79 / examples/libero/main.py |
| 模型 | 现有 pi05_libero 服务；检查点期望 gs://openpi-assets/checkpoints/pi05_libero |
| 任务 | libero_spatial 的10个任务，各1回合 |
| 动作执行 | 官方前5步再查询，7维动作直接 env.step；移动与夹爪由模型同时控制 |
| 图像/状态 | 官方原始代码的翻转、缩放和状态布局 |
| 随机种子 | 客户端7；不声称固定服务端采样随机状态 |
| 控制器 | 原生 LIBERO/robosuite 配置，保留原生限幅/缩放/碰撞 |
| 记录 | 官方入口源码、版本、模块路径、清单、控制台日志、10个视频、解析结果、文件哈希 |

脚本下载固定官方文件到该次结果目录并直接运行，无 monkey-patch、无强制张开、无额外位移限幅、无 simulator-truth 接管。
本机安装的 LIBERO、robosuite、MuJoCo和客户端依赖仍是已经适配 Spark 的环境，不宣称整套依赖是官方原始 Docker。
只运行1回合/任务也避免该版本官方视频文件名没有 episode 编号导致重复覆盖。
默认60分钟超时，失败或中断保留结果；官方日志直接写入 console.log。

另开终端查看进度：

```bash
cd /home/test/Robot/Cart-Double-Passive-Direct
find diagnostics/official_libero_baseline -name console.log -print
# 对打印出的本次路径执行 tail -f <完整路径>
```

### 判定与边界

run.json 中 completed 仅表示10回合完整、无捕获异常、进程退出正常且有10个非空视频。
实际抓取/放置表现看 metrics.successes、metrics.success_rate 和视频。0/10也可能是完整但失败的基线。
官方客户端用 done 判定成功；我们保留它的语义，不更换成自定义成功门槛。
若日志含 Caught exception 或 Traceback，即使官方程序返回0，也不能判作完整基线。
10回合是基础检查，不是复现官方全量成功率，也不是对每个任务可靠性做统计估计。
脚本不改变官方入口，所以没有新增逐动作/逐推理归档，model_inference_count 明确 unavailable。
服务健康、命令行进程候选和声明的模型名称不能证明已加载的权重身份；server.identity_status 保持 unverified，后续需结合服务启动日志核对。
非空视频数不是图像质量验证，后续仍需实际查看视频。

### 和当前 Isaac 桥接的对照顺序

1. 先核对原生环境是否完整运行、是否抓取/放置成功、模型与归一化日志是否匹配。
2. 若原生也失败，先查模型/输入/依赖，不能只调 Isaac 增益。
3. 若原生成功，而 Isaac 代理方块场景失败，说明链路/场景差异值得优先检查；不能据此单独认定某个坐标变换或控制器有错。
4. 比较时逐项列出场景对象、图像、TCP/世界映射、夹爪语义、动作缩放、控制频率、动力学和门控。当前强制张开是明确的行为差异。
5. Arena 原生 Isaac+DROID 模型属于另一条基线，不与当前 pi05_libero 成功率直接混比。

### 推送结果

```bash
git add -- diagnostics/official_libero_baseline/
git commit -m "Add official LIBERO native baseline results"
git push origin main
```

先看 run.json 的 large_files_over_95MiB。若非空，避免把超过GitHub单文件限制的视频直接提交；先提交日志/JSON/官方入口，并另行保存大视频。
若另行运行了 Arena 平台检查，可单独 git add -- diagnostics/official_arena_baseline/。

来源：[官方 LIBERO 示例](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79/examples/libero)、[Arena 平台配置](https://github.com/isaac-sim/IsaacLab-Arena/blob/3b0f10ce2df6c25649afbbf5aed0b1cd27d5cd66/pyproject.toml)、[Arena DROID 定义](https://github.com/isaac-sim/IsaacLab-Arena/blob/3b0f10ce2df6c25649afbbf5aed0b1cd27d5cd66/isaaclab_arena/embodiments/droid/droid.py)。
