# 组培 SOP 实训评估台

本仓库是 **朱顶红无菌体系建立** 的检测与识别部分（Tissue-Culture SOP Monitor）。侧面摄像头拍摄操作过程，实时识别器具与试剂、跟踪物体，并读取瓶身文字。

检测类别：瓶子、镊子、美工刀、手套、灭菌器。瓶身 OCR 归到酒精 / 无菌水 / 次氯酸钠 / 灭菌瓶 / 培养基。

## 用到的技术

| 环节                   | 技术                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| 目标检测（旋转框 OBB） | 基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的自训练权重 |
| 多目标跟踪             | ByteTrack                                                                        |
| 瓶身文字识别           | PaddleOCR（PP-OCRv6 tiny），结果与跟踪 ID 绑定                                   |

## 常用脚本

```bash
python video_track_ocr.py # 摄像头：检测 + 跟踪 + 瓶子 OCR
python video_ocr.py       # 摄像头：检测 + 瓶子 OCR
python detect.py          # 图片/视频离线检测
```

检测部分基于 Ultralytics YOLO，许可证仍为 AGPL-3.0。
