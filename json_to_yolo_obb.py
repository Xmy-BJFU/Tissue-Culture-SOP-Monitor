"""将 LabelMe / X-AnyLabeling 的 oriented_rectangle JSON 转为 YOLO-OBB txt。."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 类别顺序固定，和后续 data.yaml 保持一致
CLASS_NAMES = ["瓶子", "镊子", "美工刀", "手套", "灭菌器"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON 旋转框 → YOLO-OBB txt")
    parser.add_argument(
        "src",
        nargs="?",
        default=r"C:\Users\Administrator\Desktop\灭菌器",
        help="含图片和 json 的目录",
    )
    return parser.parse_args()


def json_to_lines(json_path: Path, name_to_id: dict[str, int]) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = float(data["imageWidth"])
    h = float(data["imageHeight"])
    if w <= 0 or h <= 0:
        raise ValueError(f"无效宽高: {json_path}")

    lines: list[str] = []
    for shape in data.get("shapes") or []:
        label = shape.get("label") or ""
        if label not in name_to_id:
            raise ValueError(f"{json_path.name} 含未登记类别「{label}」，请补进 CLASS_NAMES")
        pts = shape.get("points") or []
        if len(pts) != 4:
            raise ValueError(f"{json_path.name} 的「{label}」不是 4 个点: {len(pts)}")
        coords: list[str] = []
        for x, y in pts:
            xn = min(max(float(x) / w, 0.0), 1.0)
            yn = min(max(float(y) / h, 0.0), 1.0)
            coords.extend([f"{xn:.6f}", f"{yn:.6f}"])
        lines.append(f"{name_to_id[label]} " + " ".join(coords))
    return lines


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    if not src.is_dir():
        raise FileNotFoundError(src)

    name_to_id = {n: i for i, n in enumerate(CLASS_NAMES)}
    json_files = sorted(src.glob("*.json"))
    n_boxes = 0
    for jp in json_files:
        lines = json_to_lines(jp, name_to_id)
        jp.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_boxes += len(lines)

    (src / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    n_img = sum(1 for f in src.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"目录: {src}")
    print(f"JSON → TXT: {len(json_files)} 个文件，共 {n_boxes} 个框")
    print(f"图片: {n_img}（无 json 的图保持无 txt，作为负样本）")
    print("类别:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
