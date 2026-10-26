# 红外无人机数字变焦项目交接说明

更新日期为2026年10月15日。本项目为红外无人机的操作人员生成裁剪放大后的观看画面，当前完成官方预训练大模型的首轮比较，保留HAT和SwinIR四组权重。当前HAT观感作为主要比较基准，但尚未完成最终操作人员验收；项目处于PC画质验证阶段，也未进入新的红外适配训练或实时部署。

## 开发交付边界

用户明确要求交付给开发人员，内容只有代码、模型和测试数据，不需要观看验收资料。实际目录为`delivery/开发交接_20261015`，从`开发交接说明.txt`开始；精确清单见`delivery/交付索引.json`，包内`文件校验.json`记录SHA256。

代码包括16个项目Python文件、官方源码和必要隔离依赖、常规及可选TensorRT依赖版本；模型为4组当前保留权重；测试数据仅保留ROS bag主集1,404帧、困难集600帧及审核/几何协议。包内保留原相对目录布局，另外新建`config/protocol.json`使用相对权重路径；这是新配置，不替代冻结协议或原环境资格。官方源码内版本元数据为现有prepare入口读取提交号所需，根项目Git历史不交付。

不交付观看成果图片/视频、历史outputs、旧资格报告、LLVIP、本机环境、下载缓存或已编译TensorRT引擎。目标机器准备CUDA环境，默认先运行FP32；优化模式重新资格检查，TensorRT在目标环境重建。代码仍是PC开发原型，不是实时部署产品，第三方资产分发前核查许可。

2026-10-15按用户明确要求实际删除不再使用的数据与材料：LLVIP、airui（含开发包副本）、TensorRT安装wheel和分段缓存、旧观看归档、首轮重复PNG/视频、旧数据审计与视觉审查、3份废弃ONNX导出，以及主项目和开发包的历史交付专用核验脚本。共删除22,523个文件约8.70GiB，项目由约15.72GiB降至7.03GiB；保留首轮协议、指标和JSON报告、最新加速资格、正式TensorRT图/引擎及运行环境。最新清理记录为`outputs/project_cleanup_20261015/cleanup.json`，不再保留已删媒体的完整交付能力。

当前根目录13个Python文件、`tools/rosbag`内3个ROS数据工具、`docs`内2份方案。开发包已同步只保留ROS bag测试数据并刷新校验清单；原推理代码、现役模型、保留测试帧和资格内容未改，没有提交或推送Git。

项目根目录是 `C:/Users/Administrator/Desktop/upsample`。本说明供其他模型或工程人员接手使用。先阅读当前需求和运行方式，再查看已有结果；历史报告中的早期任务、已删除路径和训练记录不能作为当前运行状态。

## 当前需求

摄像头按用户描述不能光学变焦。现有航拍数据约在100米高度采集，后续可能飞到200米或300米，同一目标在原图中的像素会减少。处理流程是原始红外帧、固定区域裁剪、超分或上采样、输出放大后的局部画面。

当前输入和主显示输出均按640×512处理，覆盖约2倍和3倍数字变焦。2倍裁剪通常是320×256，3倍约213×171。裁剪本身保留原像素；放大后低采样、颗粒和模糊更容易显现。网络原生4倍结果后缩放到640×512，必须标注原生倍率与显示倍率的区别。

用户关心整张图中车辆、道路、建筑等目标轮廓更清楚、颗粒更少、目标更容易辨认，以及连续视频稳定。优先交付无文字、无框、无拼接的完整图片和视频，同尺寸双三次作为基线。内部裁块与指标用于分析，不代替整图交付。

当前用途已从ORB或VINS特征匹配改为操作人员观看，允许较大算力平台。RK3588、VINS跟踪分数和特征点数量均不再决定显示模型是否晋级。最终硬件和实时延迟预算还未确定。

## 当前模型

所有权重在 `C:/Users/Administrator/Desktop/upsample/third_party/display_weights`。

| 运行名称 | 权重文件 | 用途 |
|---|---|---|
| hat_real_x4 | Real_HAT_GAN_SRx4.pth | 原生4倍，支持显示2倍和3倍；较柔和的视觉增强候选 |
| swinir_real_large_x4 | 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth | 原生4倍，支持显示2倍和3倍；较锐利的视觉增强候选 |
| swinir_classic_x2 | 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth | 原生2倍，结构保真对照 |
| swinir_classic_x3 | 001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth | 原生3倍，结构保真对照 |

本轮使用官方RGB预训练权重，灰度输入复制为三通道，输出转换为灰度；没有新增红外训练。不要把GAN生成的纹理解释为真实温度、目标细节或有效分辨率。经典SwinIR保留为保真对照，不表示它在所有实拍场景都优于视觉增强版本。

官方源码目录为 `third_party/HAT`、`third_party/SwinIR` 和 `third_party/BasicSR_display`。`display_model_adapter.py`使用官方网络定义，拆开框架注册与导入；严格加载权重，未改网络前向逻辑。当前四模型共8项官方方形和非方形入口对照一致，记录在 `outputs/project_cleanup_20261009/validation/model_checks.json`。

默认使用FP32并关闭TF32。首轮完整ROI的FP16检查中，SwinIR通过，HAT出现非有限输出，因此首轮连续比较统一使用FP32。不能直接给HAT打开FP16；需要重新定位原因并通过完整ROI数值和画质检查。

2026-10-10新增独立的可选加速路线：完整开发ROI诊断的首个FP16非有限模块为`layers.5.conv`；经多时刻和困难帧精度探针后，`float16_safe`将输入卷积、六个残差组的末端卷积和输出重建卷积保留为FP32，还将首个完整残差组与全部OCAB跨窗注意力块保留为FP32；其余注意力、MLP与CAB中的可用算子使用FP16。不再仅保护溢出层。可选`sdpa`替换HAT两类注意力的执行方式，每头30通道零填充至32，保留原缩放、位置偏置和移位掩码。官方源码与权重不变，但优化模式的前向实现已经改变，元数据明确记录`core_forward_modified=true`。默认官方FP32路径不变，历史首轮记录也未改成优化版本。资格检查新增多时刻、开发/复核分组、重复计时和短连续帧运行误差检查；使用新`--out`保存资格JSON，已有文件拒绝覆盖。非默认单图/视频入口必须通过匹配当前代码、协议、权重与GPU的`--precision-checks`报告。命令见README的可选HAT推理加速章节；BF16是实验选项，未因此确认TensorRT、INT8或实时部署。

## 数据及适用范围

| 项目内路径 | 内容 | 当前用途 |
|---|---|---|
| data/rosbag_frames_v1 | 1,404张无损640×512灰度PNG及时间戳、哈希 | 当前测试直接读本地PNG，不必重跑bag |
| data/rosbag_frames_hard_v1 | 600张困难连续帧，四段各150帧 | 后续画质压力测试；是有意选择的困难开发集 |
| data/display_protocol_v1 | audit.json及三段连续复核的geometry缓存 | 冻结源数据、ROI和辅助运动几何 |

主抽帧集包含两段航拍和一段棋盘格，不能把1,404张全部算作航拍。每份航拍各有三个连续片段、每段150帧，两包共900张主连续测试帧；另外的概览帧按间隔抽取，不能拼成连续视频。棋盘格只用于几何诊断。

原始bag位于 WSL Ubuntu-22.04 的 `/home/dongcheng/nogps/data/Data0916/flight_records`，Windows可访问 `\\wsl.localhost\Ubuntu-22.04\home\dongcheng\nogps\data\Data0916\flight_records`。本地PNG直接保存UYVY中的8位Y亮度值，没有增强，也不是14位或16位辐射RAW。

开发包使用 `infrared_20260916_111549`，复核包使用 `infrared_20260916_112100`，各取clip_10pct、clip_50pct和clip_90pct。静态参考索引为15、30、45、60、75、90、105、120、135、149。两包此前都参与过实验，复核包不能称为全新独立测试集；训练应按整包或场景划分，不能随机把相邻帧分进训练和验证。

LLVIP下载数据和同事提供的airui已于2026-10-15从主项目及交付包中删除，当前测试仅使用保留的ROS bag抽帧与协议。若后续开展训练，需另行准备有明确采样、模糊和噪声假设的低清输入与对应参考；原参考不可直接称为无噪高清真值，也不应使用当前GAN输出作为真实高清标签。

现有约100米数据的裁剪放大与先降采样再恢复是两类测试。前者评估这批实拍图的数字变焦，后者评估指定退化下的恢复。两者都不能直接证明实际200米或300米画质收益；100米高度目前也是用户提供的场景描述，未逐帧核对飞行记录。

## 保留的Python代码

当前共16个项目Python脚本：根目录保留13个推理、评估及公共模块；3个ROS数据工具已归入`tools/rosbag`。公共模块会被入口导入，不能单凭没有main函数就删除。

| 文件 | 职责 |
|---|---|
| run_display_zoom.py | 日常单张图裁剪和超分，输出完整结果与基线 |
| run_hat_tensorrt.py | 独立固定尺寸ONNX导出、TensorRT构建及误差/计时对照；两尺寸两精度已完成抽样检查，未接入日常推理 |
| display_model_adapter.py | 官方架构加载、严格权重加载、模型推理和计时 |
| display_support.py | 独立裁剪、缩放、运动几何和公共指标函数 |
| ir_sr.py | 单通道图像读取、归一化及误差计算 |
| runtime_v5.py | 哈希、原子JSON写入及Windows进程锁 |
| prepare_display_models.py | 四组模型清单、新实验协议与官方权重获取；推理入口也导入模型清单 |
| evaluate_display_zoom.py | 静态crop与sampling比较，支持独立预降噪对照 |
| evaluate_display_video.py | 固定ROI的连续帧比较及时间残差诊断 |
| zoom_restoration_metrics.py | 颗粒、边缘宽度、边缘位移和光晕代理指标 |
| check_display_models.py | 严格加载与官方推理入口一致性检查 |
| check_display_precision.py | 完整开发ROI的FP32与FP16对照 |
| encode_display_videos.py | 从无损PNG生成H.264预览，并逐帧核对编码误差 |
| tools/rosbag/extract_rosbag_frames.py | WSL抽帧；当前采样计划固定为原三份bag，新录制需修改计划 |
| tools/rosbag/inspect_flight_rosbags.py | WSL检查bag及UYVY解码；抽帧脚本依赖同目录内此文件的函数 |
| tools/rosbag/verify_rosbag_frames.py | 冻结的v1本地1,404帧核验；新数据集需调整固定数量约束 |

两个首轮自动调度脚本和只含首轮固定结论的汇总脚本已移出项目。以后手动依次运行评估入口，新结果另写报告，不复制旧汇总中的结论。

## 环境与单图运行

当前Windows解释器是 `D:/Python3/python.exe`，Python 3.14.5、PyTorch 2.12.0+cu126，GPU为RTX4060 Ti 16GB。其他版本在 `requirements-tested.txt`；隔离依赖目录为 `third_party/display_deps`。运行不需要重新训练、重新下载或修改全局环境。

下面使用首轮协议中的现存权重做独立推理。它仍可用于四个保留模型，但历史协议记录了9个模型，不要重新运行其中淘汰组。

```powershell
Set-Location 'C:/Users/Administrator/Desktop/upsample'
$env:PYTHONUTF8 = '1'
$py = 'D:/Python3/python.exe'
& $py -B run_display_zoom.py --run outputs/display_zoom/run_20261008_162935 --model hat_real_x4 --input data/rosbag_frames_v1/infrared_20260916_112100/clip_90pct/frame_00075.png --scale 3 --roi 215 55 213 171 --out outputs/new_hat_x3
```

将model改为swinir_real_large_x4即可比较SwinIR。经典SwinIR名称与显示倍率必须对应。2倍车辆样张用 `--scale 2 --roi 160 40 320 256`。新图片应选自己的ROI，不要机械复用车辆样张坐标；不指定ROI时按倍率做中心裁剪。

输出目录必须不存在。目录内 `超分结果.png` 是640×512完整灰度输出，`双三次.png` 是同尺寸基线，`网络原生RGB输出.png` 保存后缩放之前的诊断图，metadata.json保存权重哈希、ROI和计时。输入为整数单通道图，默认8位；读图函数也支持声明8至16位，但当前冻结评估数据和协议是8位，不能把8位亮度图改参数后当16位RAW。

模型配置和结果元数据有绝对路径。若迁移到其他机器，先更新项目根路径、权重位置和新协议；不要修改冻结的历史记录来伪装同一实验。当前锁使用msvcrt，推理默认依赖CUDA，因此现有PC入口不能直接视为已完成Linux或CPU部署。

## 新实验的运行顺序

下面是后续复现实验的命令示例，本次清理没有启动这一整轮。new_comparison目录必须不存在；相同目录内每个阶段也默认拒绝覆盖。先完成模型检查，再运行静态评估，最后选择视频候选。

```powershell
& $py -B prepare_display_models.py --run outputs/display_zoom/new_comparison
& $py -B check_display_models.py --run outputs/display_zoom/new_comparison
& $py -B evaluate_display_zoom.py --run outputs/display_zoom/new_comparison --role development
& $py -B evaluate_display_zoom.py --run outputs/display_zoom/new_comparison --role review
```

prepare默认复用现存权重，缺失才下载，并更新active_run.txt。check_display_models会拒绝含淘汰模型的历史协议，避免覆盖原9模型检查记录。新协议只有四组模型，其静态记录数量与首轮9模型不同。

需要比较温和预降噪时另跑独立阶段；它不会成为默认推理步骤：

```powershell
& $py -B evaluate_display_zoom.py --run outputs/display_zoom/new_comparison --role development --revision _denoise --denoise --models hat_real_x4 swinir_real_large_x4
& $py -B check_display_precision.py --run outputs/display_zoom/new_comparison --models hat_real_x4 swinir_real_large_x4
& $py -B evaluate_display_video.py --run outputs/display_zoom/new_comparison --models hat_real_x4 swinir_real_large_x4 --precision float32
```

视频阶段会处理三段复核、两个倍率和两个模型，推理耗时较长。当前程序是离线评估，逐帧调用现有单帧模型，不是已实现的视频超分网络；SIFT单应性只是时间残差的运动代理。若修改协议为新场景，需同步检查程序对150帧、640×512和25fps的固定约定。

编码入口还要求新run内存在UTF-8无BOM的display_candidates.json，记录已根据开发包选择的候选，例如：

```json
{"models": ["hat_real_x4", "swinir_real_large_x4"]}
```

保存后运行：

```powershell
& $py -B encode_display_videos.py --run outputs/display_zoom/new_comparison
```

无损PNG是主比较依据；video/hq_videos中的H.264用于观看，video/videos中的初始MPEG-4不是最终高质量交付。编码门槛为每帧平均误差不超过0.75DN、最低PSNR不低于40dB，比较的是编码前后同一输出，与超分恢复精度无关。编码入口的完成计数仍针对三片段、两倍率、基线加两候选；改变候选数时需调整。

ROS读取脚本需要WSL中的rosbags环境，当前Windows推理环境未安装rosbags；它们也不属于日常单图推理依赖。现有PNG已保存，只有读取新bag时才需要它们。三个工具现从项目根以`python tools/rosbag/<脚本名>.py ...`调用，或在WSL用对应路径调用；抽帧与检查工具保持同目录，导入关系不变。

## 最新结果与结论

首轮曾交付48张640×512完整灰度PNG和18段各150帧的视频；2026-10-15已按开发用途清理这些媒体及首轮重复中间图片/视频。历史说明、ROI和交付清单仍留在`outputs/操作员显示大模型结果/run_20261008_162935`，首轮协议、指标CSV及JSON报告仍留在`outputs/display_zoom/run_20261008_162935`，不再将它们称为完整媒体交付。当前可用代表样图在`outputs/hat_acceleration_20261010`及TensorRT正式`check_x*`目录。

首轮共测试9组官方权重，开发和复核各30个静态参考时刻，再评估450个连续复核源帧；合并使用的不同源记录为480份。不同倍率、方法和重复静态时刻不增加独立样本量。本轮没有新增训练。

车辆样张中，两种视觉增强能让部分边界更利落、背景更平滑；SwinIR通常保留更强纹理，HAT较柔和。植被和不规则地面场景也可能被整理成较大的明暗块，收益未必与车辆场景一致。HAT在首轮三段、两倍率下的对齐时间残差较低，因此保留为视频观看候选；平滑本身也能降低残差，不能据此宣布更准确。当前尚未由操作人员确认最终版本。

固定温和预降噪对照的平均输出改变约0.04至0.09个灰度级，没有证明额外可见改善，首轮视频保持原始输入。不要把这一强度下的结果推广为所有降噪都无效。

首轮视觉候选在当前GPU的FP32原型约0.7至1.8秒一帧。预览25fps是播放节奏，不是推理帧率；当前不能承诺实时25Hz。首轮没有实现TensorRT部署或最终硬件优化，后续独立加速试验见下文。

2026-10-10的可选HAT保守混合精度+SDPA版本在同一RTX4060 Ti 16GB上，通过72个协议ROI及24个困难抽样ROI。每倍率54次重复计时，准备好的ROI到640×512 CPU结果中位耗时：2倍从1.742秒降至1.151秒，3倍从0.818秒降至0.524秒；排除读图、模型加载、预热、保存和编码，不是完整显示链路延迟。96个ROI的最坏单图平均显示差为0.00893DN，显示最大差1.052DN。资格文件为`outputs/hat_acceleration_20261010/qualification_v4.json`，汇总为同目录`summary.json`。四模型8项默认官方对照与54张历史整图重放也通过。困难集仅抽查四段各3帧，未全验600帧或执行完整连续视频验收。更激进约2倍加速版本出现原生误差超限，保留失败记录但不作为合格方案；没有放宽误差门槛。当前仍只有约0.87/1.91帧每秒的上述处理速度，需要后续部署优化，不能称为实时版本。优化前根目录代码、文档和必要配置已保存到本地Git提交`7a46689`；大数据、权重、输出与独立版本管理的第三方源码未纳入根仓库，均仍留在本地。

同日完成独立TensorRT试验，入口为`run_hat_tensorrt.py`，阶段为`export`、`build`、`check`，不修改已合格PyTorch适配器或接入日常推理。`third_party/tensorrt_env`复用系统Torch，TensorRT及ONNX依赖位于虚拟环境；TensorRT11.4.0.106 CUDA12运行库与Python3.14 Windows绑定均通过官方SHA256后安装。两种固定ROI尺寸、FP32与显式FP16/FP32混精共四个图和引擎均已实际构建并检查；正式图为`export_x2_fp32_v2`、`export_x3_fp32_v3`、`export_x2_mixed`、`export_x3_mixed`。实际输入NPY、源码快照、前处理、权重、图、构建与引擎哈希可追溯。固定mask/均值包装保持操作顺序，官方源码与权重不改，但执行实现改变，汇总记录`core_forward_modified=true`。TensorRT11通过显式Cast表达混精，沿用敏感模块FP32保护并导出官方注意力，不是将SDPA执行图直接搬入TensorRT；TF32关闭，没有INT8或训练。

本轮TensorRT混精同轮中位耗时：2倍从官方FP32的1.735秒、既有混精+SDPA的1.143秒降至0.750秒，分别加速2.31倍、1.52倍，约1.33fps；3倍从0.812秒、0.516秒降至0.327秒，分别加速2.49倍、1.58倍，约3.06fps。每倍率、每方法54次正式计时，TensorRT混精P95为756.9/330.2ms，口径仍是准备好的CPU ROI到CPU 640×512结果，排除读图/解码、加载、预热、保存和编码，不是相机到屏幕延迟。FP32 TensorRT中位1.060/0.488秒，相对同轮既有优化仅约1.07/1.05倍。

每种引擎精度共检查72个协议ROI与24个困难抽样ROI，来自48个不同源帧，两精度与两倍率的重复使用不增加独立源帧数。每倍率48个ROI、每候选方法84次误差检查、26个实际相邻帧对，四份检查均通过原误差门槛。TensorRT混精最坏单图平均显示差0.04250DN、显示最大差1.3122DN、原生灰度最大差2.494DN；96对完整8位PNG最大差2DN，车辆第75帧两倍率最大差1DN。查看的六张车辆/困难整图未见明显变化，但不是逐像素一致或操作人员验收。相邻帧运行误差变化通过，不代表完整视频闪烁验收；困难集仍只抽查四段各74/75/76帧，未全验600帧或整段视频。

汇总为`outputs/hat_tensorrt_20261010/summary.json`；四份`check_x*`目录含逐次误差、计时及完整PNG。`memory_probe.json`中的混精两倍率预热后driver显存差额约4130/2034MiB，不是峰值，也不能和PyTorch allocated显存直接比较；PNG量化差异见`png_comparison.json`。正式图、构建和检查的追溯记录保留；2026-10-15删除了3份早期废弃导出及安装下载缓存，历史导出/下载过程不再全量保留。现役资格、权重与正式产物内容不变。当前结论是可保留为离线加速备选，仍未完成实时显示部署；默认PyTorch入口及`qualification_v4.json`不切换、不失效。跨硬件需重新构建并检查。复现命令见README；后续实时路线先明确硬件、帧率与延迟预算，再考虑轻量原生2/3倍模型或蒸馏，尚未启动这些工作。

crop是真实固定区域放大，没有高清真值。sampling使用INTER_AREA降采样的原图作相对参考，裁去8个输出像素边界计算误差；该参考有噪声，退化也不同于经典模型训练条件。PSNR、颗粒、锐度与时间残差只能辅助判断，不能单独证明真实细节恢复或操作人员辨识提升。

## 清理与验证状态

2026年10月9日第一轮清理移出约25.46GB旧模型、训练缓存、旧环境和大体积输出，保留HAT和SwinIR四组权重、实拍数据及首轮完整交付。NAF、Restormer、ESPCN、RFDN、Real-ESRGAN、HAT-L、更锐HAT等旧权重和实验代码不再是当前可运行资产；旧模型当时的需求不同，淘汰不代表其在所有场景都无效。

2026-10-09进一步移出run_display_review_after_development.py、run_display_video_pipeline.py和summarize_display_zoom.py，当时CSV读取留在历史交付核验脚本中；该核验脚本已在2026-10-15删除。2026-10-09采用回收站移动，2026-10-15本轮则为真实删除并释放空间；旧清单在outputs/project_cleanup_20261009及outputs/project_cleanup_20261009_scripts。

历史小报告在outputs/history_reports。冻结的源码快照和旧文档属于当时记录，可能包含已删除脚本及失效引用，不应作为当前启动入口。当前根目录代码及本说明是清理后的状态；首轮历史9模型协议与原始报告保持原样。

第一轮清理当时核对了17,603个保留文件的SHA256，并通过原48张PNG、18段视频的交付核验；该结论为历史记录。2026-10-15已删除首轮媒体和专用核验脚本，不再列出旧验证命令，也不能据旧报告声称这些媒体仍完整留存。现役运行使用模型检查、精度资格检查、ROS bag帧核验和TensorRT的check阶段；本次删除前后对7,238个受保护文件核对SHA256，清单见最新cleanup.json。

## 接手后的工作顺序

先重放一张车辆图，再让操作人员比较三种完整输出和连续视频，确认更喜欢轮廓、平滑程度及能接受的纹理改变，按场景记录结论。不要只凭一张截图或某个指标宣布最终模型。

若迁移问题仍明显，再设计现存模型的红外适配训练：明确数据分组、对应参考、退化假设、损失和验收图，先恢复目标，再把较强感知增强作为独立候选。当前项目已移除旧训练入口，开始新训练需要编写针对当前任务的训练代码，不能宣称已有大模型红外训练正在运行。

若单帧有可见收益而连续视频闪烁，再评估原方案中尚未执行的BasicVSR++等视频路线，并明确是否允许等待未来帧。后续用实际更高高度数据核实收益，最后根据目标硬件和延迟预算优化部署。

开发输出使用新的outputs子目录，源图、时间戳、冻结报告与权重哈希保持可追溯。HAT和SwinIR保留权重是后续基准，调整模型、退化或精度后重新验证整图和视频。用户主要需要完整画面比较，交付说明应直接指出具体场景的改善和问题。
