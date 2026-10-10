# 红外无人机数字变焦项目交接说明

更新日期为2026年10月9日。本项目为红外无人机的操作人员生成裁剪放大后的观看画面，当前完成官方预训练大模型的首轮比较，保留HAT和SwinIR四组权重。项目处于PC画质验证阶段，尚未确定最终模型，也未进入新的红外适配训练或部署。

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

## 数据及适用范围

| 项目内路径 | 内容 | 当前用途 |
|---|---|---|
| data/airui | 测试同事实拍红外图片，包含不同显示配色 | 实拍整图观察；彩色或伪彩输入需另行处理，不能直接作为单通道辐射数据 |
| data/rosbag_frames_v1 | 1,404张无损640×512灰度PNG及时间戳、哈希 | 当前测试直接读本地PNG，不必重跑bag |
| data/rosbag_frames_hard_v1 | 600张困难连续帧，四段各150帧 | 后续画质压力测试；是有意选择的困难开发集 |
| data/LLVIP/infrared | train 12,025张、test 3,463张，合计15,488张红外JPEG | 后续适配训练的补充数据，不是本无人机高空真值 |
| data/display_protocol_v1 | audit.json及三段连续复核的geometry缓存 | 冻结源数据、ROI和辅助运动几何 |

主抽帧集包含两段航拍和一段棋盘格，不能把1,404张全部算作航拍。每份航拍各有三个连续片段、每段150帧，两包共900张主连续测试帧；另外的概览帧按间隔抽取，不能拼成连续视频。棋盘格只用于几何诊断。

原始bag位于 WSL Ubuntu-22.04 的 `/home/dongcheng/nogps/data/Data0916/flight_records`，Windows可访问 `\\wsl.localhost\Ubuntu-22.04\home\dongcheng\nogps\data\Data0916\flight_records`。本地PNG直接保存UYVY中的8位Y亮度值，没有增强，也不是14位或16位辐射RAW。

开发包使用 `infrared_20260916_111549`，复核包使用 `infrared_20260916_112100`，各取clip_10pct、clip_50pct和clip_90pct。静态参考索引为15、30、45、60、75、90、105、120、135、149。两包此前都参与过实验，复核包不能称为全新独立测试集；训练应按整包或场景划分，不能随机把相邻帧分进训练和验证。

LLVIP红外JPEG以重复RGB通道保存，当前CLI要求单通道整数图像，使用前需显式转灰度。LLVIP可见光与检测XML标注已移出项目；超分监督需要低清输入与对应参考图，不需要目标检测类别标签。若开展训练，可从参考图构建有明确采样、模糊和噪声假设的低清输入；原参考本身有噪声，不可称为无噪高清真值，也不应直接用当前GAN输出当高清标签。

现有约100米数据的裁剪放大与先降采样再恢复是两类测试。前者评估这批实拍图的数字变焦，后者评估指定退化下的恢复。两者都不能直接证明实际200米或300米画质收益；100米高度目前也是用户提供的场景描述，未逐帧核对飞行记录。

## 保留的Python代码

项目根目录保留16个脚本，各有当前用途。公共模块会被入口导入，不能单凭没有main函数就删除。

| 文件 | 职责 |
|---|---|
| run_display_zoom.py | 日常单张图裁剪和超分，输出完整结果与基线 |
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
| verify_display_round.py | 固定首轮交付的完整性核验；包含首轮计数，不能用来验证任意新实验 |
| extract_rosbag_frames.py | WSL抽帧；当前采样计划固定为原三份bag，新录制需修改计划 |
| inspect_flight_rosbags.py | WSL检查bag及UYVY解码；抽帧脚本依赖其中函数 |
| verify_rosbag_frames.py | 冻结的v1本地1,404帧核验；新数据集需调整固定数量约束 |

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

ROS读取脚本需要WSL中的rosbags环境，当前Windows推理环境未安装rosbags；它们也不属于日常单图推理依赖。现有PNG已保存，只有读取新bag时才需要它们。

## 最新结果与结论

完整交付目录是 `C:/Users/Administrator/Desktop/upsample/outputs/操作员显示大模型结果/run_20261008_162935`，包含48张640×512完整灰度PNG和18段各150帧的视频，按2倍和3倍分目录。优先对比HAT视觉增强、SwinIR视觉增强和原双三次。原始指标、原生诊断和连续输出位于 `outputs/display_zoom/run_20261008_162935`。

首轮共测试9组官方权重，开发和复核各30个静态参考时刻，再评估450个连续复核源帧；合并使用的不同源记录为480份。不同倍率、方法和重复静态时刻不增加独立样本量。本轮没有新增训练。

车辆样张中，两种视觉增强能让部分边界更利落、背景更平滑；SwinIR通常保留更强纹理，HAT较柔和。植被和不规则地面场景也可能被整理成较大的明暗块，收益未必与车辆场景一致。HAT在首轮三段、两倍率下的对齐时间残差较低，因此保留为视频观看候选；平滑本身也能降低残差，不能据此宣布更准确。当前尚未由操作人员确认最终版本。

固定温和预降噪对照的平均输出改变约0.04至0.09个灰度级，没有证明额外可见改善，首轮视频保持原始输入。不要把这一强度下的结果推广为所有降噪都无效。

视觉候选在当前GPU的FP32原型约0.7至1.8秒一帧。预览25fps是播放节奏，不是推理帧率；当前不能承诺实时25Hz。没有实现TensorRT部署或最终硬件优化。

crop是真实固定区域放大，没有高清真值。sampling使用INTER_AREA降采样的原图作相对参考，裁去8个输出像素边界计算误差；该参考有噪声，退化也不同于经典模型训练条件。PSNR、颗粒、锐度与时间残差只能辅助判断，不能单独证明真实细节恢复或操作人员辨识提升。

## 清理与验证状态

2026年10月9日第一轮清理移出约25.46GB旧模型、训练缓存、旧环境和大体积输出，保留HAT和SwinIR四组权重、实拍数据及首轮完整交付。NAF、Restormer、ESPCN、RFDN、Real-ESRGAN、HAT-L、更锐HAT等旧权重和实验代码不再是当前可运行资产；旧模型当时的需求不同，淘汰不代表其在所有场景都无效。

本次进一步移出run_display_review_after_development.py、run_display_video_pipeline.py和summarize_display_zoom.py，CSV读取功能已留在verify_display_round.py中。删除采用Windows回收站，不代表已释放磁盘空间。清单在outputs/project_cleanup_20261009及outputs/project_cleanup_20261009_scripts。

历史小报告在outputs/history_reports。冻结的源码快照和旧文档属于当时记录，可能包含已删除脚本及失效引用，不应作为当前启动入口。当前根目录代码及本说明是清理后的状态；首轮历史9模型协议与原始报告保持原样。

第一轮清理核对了17,603个保留文件的SHA256，六组HAT和SwinIR完整输出与清理前逐像素一致。本次修改CSV依赖后，再次通过原48张PNG和18段视频的交付核验。验证固定首轮使用以下命令：

```powershell
& $py -B verify_display_round.py --run outputs/display_zoom/run_20261008_162935 --out outputs/操作员显示大模型结果/run_20261008_162935 --retained-only
```

该脚本核对保留四组权重、历史18项官方检查记录和全部首轮交付，不会重新加载淘汰权重；不能据此声称当前9模型都仍可运行。更换实验规模时需要新验证逻辑。

## 接手后的工作顺序

先重放一张车辆图，再让操作人员比较三种完整输出和连续视频，确认更喜欢轮廓、平滑程度及能接受的纹理改变，按场景记录结论。不要只凭一张截图或某个指标宣布最终模型。

若迁移问题仍明显，再设计现存模型的红外适配训练：明确数据分组、对应参考、退化假设、损失和验收图，先恢复目标，再把较强感知增强作为独立候选。当前项目已移除旧训练入口，开始新训练需要编写针对当前任务的训练代码，不能宣称已有大模型红外训练正在运行。

若单帧有可见收益而连续视频闪烁，再评估原方案中尚未执行的BasicVSR++等视频路线，并明确是否允许等待未来帧。后续用实际更高高度数据核实收益，最后根据目标硬件和延迟预算优化部署。

开发输出使用新的outputs子目录，源图、时间戳、冻结报告与权重哈希保持可追溯。HAT和SwinIR保留权重是后续基准，调整模型、退化或精度后重新验证整图和视频。用户主要需要完整画面比较，交付说明应直接指出具体场景的改善和问题。
