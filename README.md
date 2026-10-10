# 红外无人机数字变焦与超分

接手本项目请先阅读[项目交接说明](PROJECT_HANDOFF.md)，包含当前需求、16个保留脚本、模型与数据、运行命令、已完成结论和后续工作。

当前用途是操作人员观看：将红外局部区域裁剪并放大为640×512，比较2倍与3倍数字变焦。选型允许较大GPU模型。当前模型均为官方预训练权重，未进行新的红外适配训练；新增输出像素不代表恢复了真实高空细节。

## 当前保留模型

| 入口名称 | 用途 | 倍率 |
|---|---|---|
| hat_real_x4 | 较柔和的视觉增强，首轮视频观看候选 | 原生4倍，后缩放；支持显示2倍、3倍 |
| swinir_real_large_x4 | 较锐利的视觉增强，整图观看候选 | 原生4倍，后缩放；支持显示2倍、3倍 |
| swinir_classic_x2 | 结构保真对照，后续红外适配参考 | 原生2倍 |
| swinir_classic_x3 | 结构保真对照，后续红外适配参考 | 原生3倍 |

尚未由操作人员确认最终版本。在RTX4060 Ti上的视觉候选约0.7～1.8秒一帧，25fps预览视频是播放速度，不能当作实时推理速度。

## 查看最新结果

[完整图片、视频与首轮说明](outputs/操作员显示大模型结果/run_20261008_162935/结果与说明.md)保留48张无标注整图和18段视频。首轮实际测试了9组权重；清理后保留其中4组，淘汰组的样张、指标及原始结论仍保留用于比较。

内部记录位于outputs/display_zoom/run_20261008_162935。历史协议中的旧代码快照及5组淘汰权重已移出项目，不能将该协议中的全部9组视为仍可运行。

## 单张图片推理

在项目根目录运行以下已逐像素核验的例子，输出目录必须是新目录：

```powershell
$env:PYTHONUTF8 = '1'
& D:/Python3/python.exe -B run_display_zoom.py --run outputs/display_zoom/run_20261008_162935 --model hat_real_x4 --input data/rosbag_frames_v1/infrared_20260916_112100/clip_90pct/frame_00075.png --scale 3 --roi 215 55 213 171 --out outputs/new_hat_x3
```

超分结果.png是无标注完整输出，双三次.png是同尺寸基线。输入必须为整数单通道灰度图；新图片应选择对应ROI。2倍车辆样张的ROI为160 40 320 256，3倍为215 55 213 171。LLVIP的JPEG虽是红外图像，但以RGB重复通道存储，需要先转单通道，不能直接传入该入口。

## 数据与代码

| 路径或入口 | 内容 |
|---|---|
| data/airui | 测试同事实拍原图 |
| data/rosbag_frames_v1 | 1,404张本地抽帧、时间戳和哈希；含900张主连续测试帧 |
| data/rosbag_frames_hard_v1 | 600张困难连续帧 |
| data/LLVIP/infrared | 15,488张红外原图，供后续训练开发；非无人机高空真值 |
| data/display_protocol_v1 | 独立保存的源数据审核和运动几何缓存 |
| run_display_zoom.py | 独立单图推理 |
| prepare_display_models.py、check_display_models.py | 新实验协议、4组模型准备和官方入口校验 |
| evaluate_display_zoom.py、evaluate_display_video.py | 静态与连续帧评估 |
| encode_display_videos.py | 视频编码和压缩质量检查 |
| display_support.py、ir_sr.py、runtime_v5.py | 独立公共函数，不依赖旧模型或训练代码 |
| extract_rosbag_frames.py、inspect_flight_rosbags.py | WSL中读取原始ROS2 bag；现有测试直接用本地PNG即可 |

当前使用D:/Python3/python.exe的CUDA环境。隔离依赖仍在third_party/display_deps，官方源码在third_party/HAT、SwinIR、BasicSR_display，权重在third_party/display_weights。版本见requirements-tested.txt；本次没有修改或安装全局环境。rosbags用于WSL抽帧，未安装进当前PC推理环境。

prepare_display_models.py为新实验生成4模型协议，默认复用现有权重；请勿用历史run目录作为新实验输出。首轮自动调度与固定结论汇总脚本已移除，后续按交接说明手动依次运行评估入口。新实验的结果报告需依据实际输出重新编写。

历史首轮交付的清理后验证命令：

```powershell
& D:/Python3/python.exe -B verify_display_round.py --run outputs/display_zoom/run_20261008_162935 --out outputs/操作员显示大模型结果/run_20261008_162935 --retained-only
```

此验证核对保留权重及全部历史交付，不重跑淘汰模型。

## 2026-10-09清理记录

旧NAF、Restormer、ESPCN、RFDN等实验代码和权重、5组本轮淘汰权重、旧CPU环境、重复训练缓存、LLVIP可见光与检测标注、来源未确认的fusion数据及旧大体积输出已移入Windows回收站。原始WSL bag未改动。历史小报告归档于outputs/history_reports；报告中提到的旧路径不保证存在。

[清理报告](outputs/project_cleanup_20261009/清理报告.md)记录删除清单、保留文件哈希、官方入口校验及整图重放。采用回收站删除，项目变小；磁盘空间需要清空回收站后才释放。

[当前显示方案](DISPLAY_ZOOM_PLAN.md)和[需求说明](DIGITAL_ZOOM_REQUIREMENTS.md)保留后续方向。


2026-10-09进一步清理三个首轮辅助脚本，根目录现保留16个有明确用途的Python文件。verify_display_round.py的CSV读取已独立，不再依赖旧汇总脚本；详情见[本次脚本清理记录](outputs/project_cleanup_20261009_scripts/清理记录.md)。
