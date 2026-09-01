"""将扁平目录中的图片+YOLO-OBB txt 按 8:2 划分到 images/labels 的 train、val。."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["瓶子", "镊子", "美工刀", "手套", "灭菌器"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分 YOLO-OBB 训练集")
    parser.add_argument("--src", default=r"C:\Users\Administrator\Desktop\300")
    parser.add_argument("--dst", default=r"C:\Users\Administrator\Desktop\data")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_classes(txt: Path) -> tuple[int, ...]:
    if not txt.exists():
        return ()
    ids = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(line.split()[0]))
    return tuple(sorted(set(ids)))


def split_group(items: list[Path], ratio: float) -> tuple[list[Path], list[Path]]:
    items = list(items)
    random.shuffle(items)
    n = len(items)
    if n == 0:
        return [], []
    if n == 1:
        return items, []
    n_train = round(n * ratio)
    n_train = min(max(n_train, 1), n - 1)
    return items[:n_train], items[n_train:]


def copy_pair(im: Path, split: str, dst: Path) -> None:
    img_dir = dst / "images" / split
    lbl_dir = dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(im, img_dir / im.name)
    lb = im.with_suffix(".txt")
    if lb.exists():
        shutil.copy2(lb, lbl_dir / lb.name)


def write_yaml(dst: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path = dst.resolve().as_posix()
    text = f"path: {path}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_NAMES)}\nnames:\n{names}\n"
    (dst / "data.yaml").write_text(text, encoding="utf-8")


def count_split(dst: Path, split: str) -> dict[str, int]:
    img_n = len(list((dst / "images" / split).glob("*")))
    lbl_n = len(list((dst / "labels" / split).glob("*.txt")))
    box_n = 0
    cls_n = defaultdict(int)
    for t in (dst / "labels" / split).glob("*.txt"):
        for line in t.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid = int(line.split()[0])
                box_n += 1
                cls_n[cid] += 1
    return {"images": img_n, "labels": lbl_n, "boxes": box_n, "cls": dict(cls_n)}


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"未找到图片: {src}")

    groups: dict[tuple[int, ...], list[Path]] = defaultdict(list)
    for im in images:
        groups[read_classes(im.with_suffix(".txt"))].append(im)

    train, val = [], []
    for key, items in sorted(groups.items(), key=lambda x: (len(x[0]) == 0, x[0])):
        tr, va = split_group(items, args.ratio)
        train.extend(tr)
        val.extend(va)

    for split, items in ("train", train), ("val", val):
        for im in items:
            copy_pair(im, split, dst)

    write_yaml(dst)
    (dst / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    print(f"源: {src}  共 {len(images)} 张")
    print(f"目标: {dst}")
    print(f"划分: train {len(train)} / val {len(val)}  (seed={args.seed})")
    for split in ("train", "val"):
        st = count_split(dst, split)
        print(f"\n[{split}] 图片 {st['images']}  标签文件 {st['labels']}  框 {st['boxes']}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i} {name}: {st['cls'].get(i, 0)}")
    print(f"\n已写入 {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
