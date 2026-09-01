# 组培 SOP 实训评估台

本仓库是 **朱顶红无菌体系建立** 的实训评估系统（Tissue-Culture SOP Monitor）。侧面摄像头拍摄操作过程，实时识别器具与试剂、跟踪物体、读取瓶身文字、估计手部关键点，再按 SOP 规则判定动作、计时浸泡，并给出四维评分和实训报告。

主要覆盖：防护检查 → 工具灭菌 → 酒精浸泡 → 无菌水冲洗 → 次氯酸钠消毒 → 无菌水冲洗×3 → 切割种球 → 斜插接种。

## 用到的技术

| 环节 | 技术 |
| --- | --- |
| 目标检测（旋转框 OBB） | 基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的自训练权重 |
| 多目标跟踪 | ByteTrack |
| 瓶身文字识别 | PaddleOCR（PP-OCRv6 tiny），结果与跟踪 ID 绑定 |
| 手部检测 / 21 点姿态 | RTMDet + RTMPose（rtmlib / OpenMMLab） |
| 动作判定 | 几何规则（倾角、持物、刀心跨度等），不训练行为模型 |
| 后端 | Python、FastAPI、Uvicorn、OpenCV |
| 前端 | HTML / CSS / JavaScript |

戴手套时在 YOLO「手套」框内估点；不戴手套时用 RTMDet 全图找手再估点。

## 启动评估台

```bash
python lab_web.py
```

浏览器打开终端提示的地址（默认 `http://127.0.0.1:7860`）。命令行离线调试可用 `video_track_ocr_hand_pose.py`。

检测部分基于 Ultralytics YOLO，许可证仍为 AGPL-3.0。
