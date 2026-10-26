# 红外无人机数字变焦与超分

接手本项目请先阅读[项目交接说明](PROJECT_HANDOFF.md)，包含当前需求、16个脚本（根目录13个，ROS数据工具3个位于`tools/rosbag`）、模型与数据、运行命令、已完成结论和后续工作。

## 开发人员交付

**交付内容仅为代码、模型和测试数据**，目录为`delivery/开发交接_20261015`。入口是[开发交接说明](delivery/开发交接_20261015/开发交接说明.txt)，精确范围见[交付索引](delivery/交付索引.json)。为保持现有导入与加载关系，包内保留原相对目录布局，不强行拆成三个互不兼容的目录。

| 类别 | 包内位置 | 内容 |
|---|---|---|
| 代码 | 根目录`*.py`、`tools/rosbag`、`third_party/HAT`/`SwinIR`/`BasicSR_display`/`display_deps` | 16个项目脚本、官方源码及必要隔离依赖，包含PyTorch和TensorRT开发入口 |
| 模型 | `third_party/display_weights` | HAT真实超分、SwinIR真实超分大模型、经典SwinIR x2/x3，共4组权重 |
| 测试数据 | `data/rosbag_frames_v1`、`rosbag_frames_hard_v1`、`display_protocol_v1` | 1,404张主测试帧、600张困难帧、审核及几何缓存，仅保留ROS bag相关数据 |
| 代码配套配置 | `config/protocol.json`、requirements、`开发交接说明.txt`、`文件校验.json` | 新建四模型相对路径配置、环境版本、运行命令和SHA256清单 |

本包**不含**观看成果图、成果视频、历史outputs、旧资格报告、LLVIP训练补充数据、原项目Git主仓库、虚拟环境、TensorRT下载缓存或本机引擎。官方源码保留各自Git版本元数据，因为现有prepare入口会读取其提交号。

从交接包根目录执行说明中的命令；新配置不依赖本机绝对权重路径，默认使用官方FP32。目标机器需要建立CUDA Python环境；加速模式需重新做资格检查，TensorRT需在目标环境另行构建。本交付不是实时部署产品，第三方源码、权重与数据外部分发前需核查许可。

2026-10-15已按ROS bag测试用途实际清理：删除LLVIP与airui（含开发包副本）、TensorRT安装/分段下载缓存、旧观看归档、首轮重复PNG/视频和过期审计、3份废弃ONNX导出，以及历史交付专用核验脚本。共删除22,523个文件约8.70GiB，项目由约15.72GiB降至7.03GiB。清单见[最新清理记录](outputs/project_cleanup_20261015/cleanup.json)。

保留ROS bag主/困难帧及测试协议、4组模型、现役代码、首轮协议与指标JSON/CSV、最终加速资格、正式TensorRT图和引擎、已安装环境。原始WSL bag未动，开发包已同步清理并刷新校验。此后不能再用已删首轮媒体做整套历史交付重验；后续测试从保留的bag帧生成新输出。

当前用途是操作人员观看：将红外局部区域裁剪并放大为640×512，比较2倍与3倍数字变焦。选型允许较大GPU模型。当前模型均为官方预训练权重，未进行新的红外适配训练；新增输出像素不代表恢复了真实高空细节。

## 当前保留模型

| 入口名称 | 用途 | 倍率 |
|---|---|---|
| hat_real_x4 | 较柔和的视觉增强，首轮视频观看候选 | 原生4倍，后缩放；支持显示2倍、3倍 |
| swinir_real_large_x4 | 较锐利的视觉增强，整图观看候选 | 原生4倍，后缩放；支持显示2倍、3倍 |
| swinir_classic_x2 | 结构保真对照，后续红外适配参考 | 原生2倍 |
| swinir_classic_x3 | 结构保真对照，后续红外适配参考 | 原生3倍 |

尚未由操作人员确认最终版本。在RTX4060 Ti上，首轮默认FP32视觉候选约0.7～1.8秒一帧；后续可选加速实测见下文。25fps预览视频是播放速度，不能当作实时推理速度。

## 查看最新结果

[首轮历史说明](outputs/操作员显示大模型结果/run_20261008_162935/结果与说明.md)及原始指标继续保留；原48张整图、18段视频及首轮大量重复中间媒体已于2026-10-15删除。当前代表样图和加速前后对照仍在`outputs/hat_acceleration_20261010`与`outputs/hat_tensorrt_20261010/check_x*`。首轮实际测试9组权重，目前保留其中4组。

内部记录位于outputs/display_zoom/run_20261008_162935。历史协议中的旧代码快照及5组淘汰权重已移出项目，不能将该协议中的全部9组视为仍可运行。

## 单张图片推理

在项目根目录运行以下已逐像素核验的例子，输出目录必须是新目录：

```powershell
$env:PYTHONUTF8 = '1'
& D:/Python3/python.exe -B run_display_zoom.py --run outputs/display_zoom/run_20261008_162935 --model hat_real_x4 --input data/rosbag_frames_v1/infrared_20260916_112100/clip_90pct/frame_00075.png --scale 3 --roi 215 55 213 171 --out outputs/new_hat_x3
```

超分结果.png是无标注完整输出，双三次.png是同尺寸基线。输入必须为整数单通道灰度图；新图片应选择对应ROI。2倍车辆样张的ROI为160 40 320 256，3倍为215 55 213 171；其他场景按协议使用各自ROI。下载及同事数据已移除，当前测试输入来自ROS bag灰度抽帧。

## 可选HAT推理加速

默认入口仍为官方FP32，旧命令不变。新增`float16_safe`把输入卷积、六个残差组的末端卷积和输出重建卷积保留为FP32；困难帧探针还要求首个完整残差组与所有OCAB跨窗注意力块保留FP32。其余注意力、MLP与CAB中的可用算子使用FP16。此策略处理了`layers.5.conv`溢出，并保护精度敏感特征；`sdpa`用PyTorch高效注意力内核执行HAT注意力，保留原权重、窗口、相对位置偏置和移位掩码。每头30通道零填充至32以满足内核要求，仍使用原注意力缩放。此模式改变了推理实现，元数据会明确记录`core_forward_modified=true`，不能称为未经修改的官方前向。

非默认模式必须先通过当前推理代码、资格检查代码、官方架构/helper/YAML、权重、协议和GPU环境对应的独立资格检查；不得使用历史FP16记录代替。下面创建新的检查产物，不覆盖首轮报告：

```powershell
& D:/Python3/python.exe -B check_display_precision.py --run outputs/display_zoom/run_20261008_162935 --models hat_real_x4 --precisions float16_safe --attention-backends sdpa --roles development review --indices 30 75 135 --consecutive 4 --repeats 3 --warmup 1 --out outputs/hat_speed_check/qualification.json
```

原生灰度与640×512显示结果相对官方FP32的误差门槛为平均≤0.10DN、P99≤0.5DN、最大≤3DN。每次计时重复均检查误差，资格入口按当前门槛复核数值而不只信`passed`标记。相邻帧运行误差变化也单独检查，但不是运动补偿闪烁评价，更不是完整视频验收；非默认视频入口至少要求每个复核ROI有3个实际相邻帧对。资格报告和图片目录均拒绝覆盖；修改被绑定的代码、官方依赖、协议、权重或环境后应重新检查。

通过后运行：

```powershell
& D:/Python3/python.exe -B run_display_zoom.py --run outputs/display_zoom/run_20261008_162935 --model hat_real_x4 --input data/rosbag_frames_v1/infrared_20260916_112100/clip_90pct/frame_00075.png --scale 3 --roi 215 55 213 171 --precision float16_safe --attention-backend sdpa --precision-checks outputs/hat_speed_check/qualification.json --out outputs/new_hat_fast_x3
```

视频评估入口也接受上述三个运行参数，但应使用新的实验目录；未因此自动启动整轮视频评估。BF16仅提供独立测试能力，不作为默认推荐；未经保护的HAT FP16仍可能溢出。TensorRT另有下述独立试验入口，已完成独立抽样检查，未接入日常推理；未进行INT8或实时部署。无损完整输出和资格记录是当前PyTorch加速方案的核查依据。

2026-10-10本机最终资格报告为`outputs/hat_acceleration_20261010/qualification_v4.json`，可直接用于当前代码的上述`--precision-checks`参数。汇总见同目录`summary.json`，完整车辆样图在`cli_fp32`与`cli_optimized`。最终72个协议ROI与24个困难抽样ROI均通过，四模型8项默认官方入口对照通过，54张默认FP32整图与历史输出逐像素一致。96个ROI的最坏单图平均显示误差为0.00893DN，显示最大误差1.052DN；车辆样图8位PNG最大差1DN。困难集只抽查四段各3帧，不是600帧全验，也未完成整段视频观看验收。

RTX4060 Ti 16GB上的重复实测中位耗时（每倍率54次计时；已加载、已预热；从准备好的灰度ROI到CPU 640×512结果，排除读图、保存、编码）为：

| 显示倍率 | 官方FP32 | 保守混合精度+SDPA | 加速比 |
|---|---:|---:|---:|
| 2倍 | 1.742秒/帧 | 1.151秒/帧 | 1.51倍 |
| 3倍 | 0.818秒/帧 | 0.524秒/帧 | 1.56倍 |

峰值PyTorch allocated显存分别从3205/1591MiB降为1695/863MiB，不是整卡总占用。更激进的约2倍加速配置曾出现原生最大误差超限，未作为当前合格版本；失败报告保留追溯。当前仍不足实时显示，不能把此结果称为完成部署。优化前本地Git基线为`7a46689`，数据、权重、输出及独立版本管理的第三方源码未纳入根仓库。

## 独立TensorRT试验（已完成本轮抽样验证）

2026-10-10已完成两种固定尺寸、FP32与显式FP16/FP32混合精度共四个TensorRT引擎的构建和实际检查。汇总位于`outputs/hat_tensorrt_20261010/summary.json`，各引擎的逐次误差、计时及完整图片位于`check_x2_fp32`、`check_x3_fp32`、`check_x2_mixed`和`check_x3_mixed`。**这是独立离线加速试验，仍不足实时显示，未接入日常单图或视频入口。**

`run_hat_tensorrt.py`提供`export`、`build`、`check`三个阶段。固定输入分别为320×256和213×171灰度ROI；保留原HAT权重、灰度复制RGB、反射补边、原生4倍裁剪、逐RGB通道clip后转亮度及CPU PIL缩放到640×512。导出包装缓存官方窗口mask，并用固定均值缓冲区替代官方前向属性重写；官方源码和权重不改，但编译及包装改变执行实现，汇总明确记录`core_forward_modified=true`。

独立环境是`third_party/tensorrt_env/Scripts/python.exe`，以`--system-site-packages`复用现有Torch等包，TensorRT与ONNX依赖安装在该虚拟环境，不改主推理环境。TensorRT11.4.0.106 CUDA12运行库和Python3.14 Windows绑定均已通过官方SHA256后安装。TensorRT11使用强类型图，混合精度通过ONNX显式Cast表达，不使用旧版全局FP16旗标；TF32关闭。混合图沿用`float16_safe`的敏感模块FP32保护，但导出的是官方注意力实现，不是PyTorch的SDPA图。未进行INT8、换权重或新增训练。

### RTX4060 Ti 16GB同轮实测

下表取混合精度TensorRT检查中的`process_ms`中位数，每倍率、每方法54次正式计时。口径为已加载、已预热的准备好CPU灰度ROI到CPU 640×512显示结果，排除读图/解码、加载、预热、保存和编码；不是完整相机到屏幕延迟。

| 显示倍率 | 官方PyTorch FP32 | 原PyTorch混精+SDPA | TensorRT混精 | 相对FP32加速 | 相对原优化加速 | 处理帧率 |
|---|---:|---:|---:|---:|---:|---:|
| 2倍 | 1.735秒 | 1.143秒 | 0.750秒 | 2.31倍 | 1.52倍 | 1.33fps |
| 3倍 | 0.812秒 | 0.516秒 | 0.327秒 | 2.49倍 | 1.58倍 | 3.06fps |

TensorRT混精P95分别为756.9ms和330.2ms。单纯FP32 TensorRT的中位耗时为1.060秒和0.488秒，相对其同轮PyTorch混精+SDPA仅约1.07倍和1.05倍，主要新增收益来自显式混合精度。上方PyTorch优化章节仍保留其第一轮独立实测，不能混用不同轮次的分母计算本轮加速比。

### 画质、覆盖与资源边界

- 每种引擎精度跨两倍率检查72个协议ROI及24个困难抽样ROI，共96个ROI，来自48个不同源帧；两精度复用同一批源帧，不增加独立样本量。每倍率检查36个协议ROI和12个困难ROI；18个主ROI各预热1次、正式重复3次，每个候选方法共84次误差检查，包含26个实际相邻帧对。
- 原生灰度及显示误差仍要求平均≤0.10DN、P99≤0.5DN、最大≤3DN，每次正式重复均检查。四份报告全部通过；TensorRT混精最坏单图平均显示差0.04250DN、显示最大差1.3122DN、原生灰度最大差2.494DN，未放宽门槛。相邻帧运行误差变化也通过平均≤0.20DN、P99≤1DN、最大≤6DN，但不是运动补偿闪烁评价。
- 96对完整8位PNG相对同轮官方FP32最大差2DN；车辆第75帧两倍率最大差1DN。查看的六张完整车辆/困难场景图未见明显观感变化，但结果不是逐像素一致，也不能代替操作人员验收。
- 困难数据仅抽查四段各74/75/76帧，没有全验600帧、整段连续视频或独立新场景。ONNX结构检查与导出前包装一致性不能替代引擎检查；本轮结论依据实际引擎输出。
- `memory_probe.json`记录一次预热后的CUDA driver可用显存差额，混精两倍率约4130/2034MiB，不是运行峰值。差额包括TensorRT、Torch I/O与背景变动，不能和前文PyTorch allocated显存直接作节省比例比较；`png_comparison.json`保存完整PNG差异统计。

### 复现实验

以下示例重新创建3倍混精图、引擎和检查。三个输出目录均须不存在，已有实验和失败记录不覆盖：

```powershell
$env:PYTHONUTF8 = '1'
$trtpy = 'third_party/tensorrt_env/Scripts/python.exe'
$run = 'outputs/display_zoom/run_20261008_162935'
& $trtpy -B run_hat_tensorrt.py export --run $run --input data/rosbag_frames_v1/infrared_20260916_112100/clip_90pct/frame_00075.png --scale 3 --roi 215 55 213 171 --precision float16_safe --out outputs/hat_tensorrt_new/export_x3_mixed
& $trtpy -B run_hat_tensorrt.py build --run $run --graph outputs/hat_tensorrt_new/export_x3_mixed --out outputs/hat_tensorrt_new/engine_x3_mixed
& $trtpy -B run_hat_tensorrt.py check --run $run --engine outputs/hat_tensorrt_new/engine_x3_mixed --precision-checks outputs/hat_acceleration_20261010/qualification_v4.json --indices 30 75 135 --consecutive 4 --repeats 3 --warmup 1 --hard-samples --out outputs/hat_tensorrt_new/check_x3_mixed
```

2倍导出改为`--scale 2 --roi 160 40 320 256`并为三个阶段使用新的`x2`目录；FP32导出改为`--precision float32`并使用新的`fp32`目录。正式历史图为`export_x2_fp32_v2`、`export_x3_fp32_v3`、`export_x2_mixed`和`export_x3_mixed`，对应四个`engine_x*`目录。

导出保存实际输入NPY、源码快照和前处理/模型/图哈希；构建保存诊断、源码快照和引擎哈希；检查绑定构建、引擎、协议及原PyTorch资格。同机固定尺寸试验不能直接视为其他硬件部署，迁移后需重新构建和检查。现有`qualification_v4.json`与默认官方FP32路径保持有效。当前可保留TensorRT混精作为离线加速备选；实时方向仍需先明确目标硬件、帧率与延迟预算，再评估更轻的原生2/3倍模型或蒸馏，不能承诺INT8单独补齐差距。

## 数据与代码

| 路径或入口 | 内容 |
|---|---|
| data/rosbag_frames_v1 | 1,404张本地抽帧、时间戳和哈希；含900张主连续测试帧 |
| data/rosbag_frames_hard_v1 | 600张困难连续帧 |
| data/display_protocol_v1 | 独立保存的源数据审核和运动几何缓存 |
| run_display_zoom.py | 独立单图推理 |
| run_hat_tensorrt.py | 独立固定尺寸导出、构建和对照试验，已完成独立抽样检查，未接入日常推理 |
| prepare_display_models.py、check_display_models.py | 新实验协议、4组模型准备和官方入口校验 |
| evaluate_display_zoom.py、evaluate_display_video.py | 静态与连续帧评估 |
| encode_display_videos.py | 视频编码和压缩质量检查 |
| display_support.py、ir_sr.py、runtime_v5.py | 独立公共函数，不依赖旧模型或训练代码 |
| tools/rosbag/*.py | WSL读取/抽取ROS2 bag及本地帧核验；现有推理直接用本地PNG即可 |

当前使用D:/Python3/python.exe的CUDA环境。隔离依赖仍在third_party/display_deps，官方源码在third_party/HAT、SwinIR、BasicSR_display，权重在third_party/display_weights。版本见requirements-tested.txt；本次没有修改或安装全局环境。rosbags用于WSL抽帧，未安装进当前PC推理环境。

prepare_display_models.py为新实验生成4模型协议，默认复用现有权重；请勿用历史run目录作为新实验输出。首轮自动调度与固定结论汇总脚本已移除，后续按交接说明手动依次运行评估入口。新实验的结果报告需依据实际输出重新编写。

旧首轮交付的专用核验入口及对应媒体已删除，不再提供该历史验证命令。现役测试使用`check_display_models.py`、`check_display_precision.py`、`tools/rosbag/verify_rosbag_frames.py`和TensorRT的`check`阶段；新输出保存到新的实验目录。

## 2026-10-09清理记录

旧NAF、Restormer、ESPCN、RFDN等实验代码和权重、5组本轮淘汰权重、旧CPU环境、重复训练缓存、LLVIP可见光与检测标注、来源未确认的fusion数据及旧大体积输出已移入Windows回收站。原始WSL bag未改动。历史小报告归档于outputs/history_reports；报告中提到的旧路径不保证存在。

[清理报告](outputs/project_cleanup_20261009/清理报告.md)记录删除清单、保留文件哈希、官方入口校验及整图重放。采用回收站删除，项目变小；磁盘空间需要清空回收站后才释放。

[当前显示方案](docs/DISPLAY_ZOOM_PLAN.md)和[需求说明](docs/DIGITAL_ZOOM_REQUIREMENTS.md)保留后续方向。


2026-10-09进一步清理三个首轮辅助脚本，根目录当时保留16个Python文件；当时的历史交付核验脚本已在2026-10-15删除。旧检查仅描述执行当时的状态，详情见[历史脚本清理记录](outputs/project_cleanup_20261009_scripts/清理记录.md)。
