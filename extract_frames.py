"""从视频抽帧，供 YOLO-OBB 标注使用。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从视频按时间间隔抽帧")
    parser.add_argument(
        "video",
        nargs="?",
        type=str,
        default=r"C:\Users\Administrator\Pictures\Camera Roll\WIN_20260827_15_29_51_Pro.mp4",
        help="视频文件路径，可省略，默认使用代码里的路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=r"C:\Users\Administrator\Desktop\1",
        help="图片输出目录，默认 images/raw_frames",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.8,
        help="抽帧间隔（秒）。默认 0.5，即每 0.5 秒保存 1 张",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多保存多少张，0 表示不限制",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="保存格式，默认 jpg",
    )
    return parser.parse_args()


def _imwrite(path: Path, frame, ext: str) -> bool:
    """OpenCV 的 imwrite 无法处理中文路径，改用 imencode 再写文件。"""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == "jpg" else []
    ok, buf = cv2.imencode(f".{ext}", frame, params)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_sec: float,
    max_frames: int,
    ext: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total / fps) if fps > 0 else 0.0
    step = max(1, int(round(fps * interval_sec))) if fps > 0 else 1
    expected = (total // step) if total > 0 else 0

    print(f"视频: {video_path}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps:.2f} fps，总帧数: {total}，时长: {duration:.1f} 秒")
    print(f"抽帧间隔: {interval_sec} 秒（约每 {step} 帧保存 1 张）")
    print(f"预计张数: {expected}")
    if fps > 0 and interval_sec < 1.0 / fps:
        print("提示: 间隔小于 1 帧时长，将退化为逐帧保存，相邻图会高度重复。")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    saved = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            name = f"{stem}_{saved:06d}.{ext}"
            out_path = output_dir / name
            if not _imwrite(out_path, frame, ext):
                raise RuntimeError(f"保存失败: {out_path}")
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        idx += 1

    cap.release()
    print(f"完成，共保存 {saved} 张 -> {output_dir.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")
    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        interval_sec=args.interval,
        max_frames=args.max_frames,
        ext=args.ext,
    )
